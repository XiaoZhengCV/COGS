#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kMaxK = 1024;

constexpr int next_power_of_two(int value) {
  int result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

__device__ __forceinline__ bool pair_less(
    const float lhs_distance,
    const int32_t lhs_index,
    const float rhs_distance,
    const int32_t rhs_index) {
  return lhs_distance < rhs_distance ||
      (lhs_distance == rhs_distance && lhs_index < rhs_index);
}

template <typename scalar_t>
__device__ __forceinline__ float squared_distance(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ reference,
    const int64_t dimensions) {
  float distance = 0.0f;
  if (dimensions == 3) {
#pragma unroll
    for (int dimension = 0; dimension < 3; ++dimension) {
      const float difference = static_cast<float>(query[dimension]) -
          static_cast<float>(reference[dimension]);
      distance = fmaf(difference, difference, distance);
    }
    return distance;
  }

  for (int64_t dimension = 0; dimension < dimensions; ++dimension) {
    const float difference = static_cast<float>(query[dimension]) -
        static_cast<float>(reference[dimension]);
    distance = fmaf(difference, difference, distance);
  }
  return distance;
}

template <int SORT_SIZE>
__device__ __forceinline__ void warp_bitonic_sort(
    float* distances,
    int32_t* indices,
    const int lane) {
#pragma unroll
  for (int sequence_size = 2; sequence_size <= SORT_SIZE;
       sequence_size <<= 1) {
#pragma unroll
    for (int stride = sequence_size >> 1; stride > 0; stride >>= 1) {
      for (int position = lane; position < SORT_SIZE;
           position += kWarpSize) {
        const int partner = position ^ stride;
        if (partner <= position) {
          continue;
        }
        const bool ascending = (position & sequence_size) == 0;
        const float lhs_distance = distances[position];
        const float rhs_distance = distances[partner];
        const int32_t lhs_index = indices[position];
        const int32_t rhs_index = indices[partner];
        const bool swap = ascending
            ? pair_less(rhs_distance, rhs_index, lhs_distance, lhs_index)
            : pair_less(lhs_distance, lhs_index, rhs_distance, rhs_index);
        if (swap) {
          distances[position] = rhs_distance;
          distances[partner] = lhs_distance;
          indices[position] = rhs_index;
          indices[partner] = lhs_index;
        }
      }
      __syncwarp();
    }
  }
}

// Small K keeps a private sorted list per lane and merges the 32 lists once.
template <int K, typename scalar_t, typename output_index_t>
__global__ void small_knn_kernel(
    float* __restrict__ output_distances,
    output_index_t* __restrict__ output_indices,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ reference,
    const int64_t query_points,
    const int64_t reference_points,
    const int64_t dimensions,
    const int64_t total_queries) {
  constexpr int kSortSize = next_power_of_two(kWarpSize * K);
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t query_linear =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (query_linear >= total_queries) {
    return;
  }

  extern __shared__ unsigned char shared_raw[];
  float* shared_distances = reinterpret_cast<float*>(shared_raw);
  int32_t* shared_indices = reinterpret_cast<int32_t*>(
      shared_distances + kWarpsPerBlock * kSortSize);
  float* warp_distances = shared_distances + warp * kSortSize;
  int32_t* warp_indices = shared_indices + warp * kSortSize;

  float local_distances[K];
  int32_t local_indices[K];
#pragma unroll
  for (int neighbor = 0; neighbor < K; ++neighbor) {
    local_distances[neighbor] = INFINITY;
    local_indices[neighbor] = std::numeric_limits<int32_t>::max();
  }

  const int64_t batch = query_linear / query_points;
  const scalar_t* query_point = query + query_linear * dimensions;
  const scalar_t* reference_batch =
      reference + batch * reference_points * dimensions;

  for (int64_t reference_index = lane;
       reference_index < reference_points;
       reference_index += kWarpSize) {
    const float distance = squared_distance(
        query_point,
        reference_batch + reference_index * dimensions,
        dimensions);
    const int32_t index = static_cast<int32_t>(reference_index);

    int insertion_position = K;
#pragma unroll
    for (int neighbor = 0; neighbor < K; ++neighbor) {
      if (pair_less(
              distance,
              index,
              local_distances[neighbor],
              local_indices[neighbor])) {
        insertion_position = neighbor;
        break;
      }
    }
    if (insertion_position < K) {
#pragma unroll
      for (int neighbor = K - 1; neighbor > 0; --neighbor) {
        if (neighbor > insertion_position) {
          local_distances[neighbor] = local_distances[neighbor - 1];
          local_indices[neighbor] = local_indices[neighbor - 1];
        }
      }
      local_distances[insertion_position] = distance;
      local_indices[insertion_position] = index;
    }
  }

#pragma unroll
  for (int neighbor = 0; neighbor < K; ++neighbor) {
    const int position = lane * K + neighbor;
    warp_distances[position] = local_distances[neighbor];
    warp_indices[position] = local_indices[neighbor];
  }
  for (int position = lane + kWarpSize * K;
       position < kSortSize;
       position += kWarpSize) {
    warp_distances[position] = INFINITY;
    warp_indices[position] = std::numeric_limits<int32_t>::max();
  }
  __syncwarp();

  warp_bitonic_sort<kSortSize>(warp_distances, warp_indices, lane);
  if (lane < K) {
    const int64_t output_position = query_linear * K + lane;
    output_distances[output_position] = warp_distances[lane];
    output_indices[output_position] =
        static_cast<output_index_t>(warp_indices[lane]);
  }
}

// The model's hot K=32 path merges 32 new references into the current top-K
// in shared memory. One warp owns one query and never writes an Nq x Nr tile.
template <typename scalar_t, typename output_index_t>
__global__ void warp_merge_knn_kernel(
    float* __restrict__ output_distances,
    output_index_t* __restrict__ output_indices,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ reference,
    const int64_t query_points,
    const int64_t reference_points,
    const int64_t dimensions,
    const int k,
    const int64_t total_queries) {
  constexpr int kSortSize = 64;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t query_linear =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (query_linear >= total_queries) {
    return;
  }

  extern __shared__ unsigned char shared_raw[];
  float* shared_distances = reinterpret_cast<float*>(shared_raw);
  int32_t* shared_indices = reinterpret_cast<int32_t*>(
      shared_distances + kWarpsPerBlock * kSortSize);
  float* distances = shared_distances + warp * kSortSize;
  int32_t* indices = shared_indices + warp * kSortSize;

  for (int position = lane; position < kSortSize; position += kWarpSize) {
    distances[position] = INFINITY;
    indices[position] = std::numeric_limits<int32_t>::max();
  }
  __syncwarp();

  const int64_t batch = query_linear / query_points;
  const scalar_t* query_point = query + query_linear * dimensions;
  const scalar_t* reference_batch =
      reference + batch * reference_points * dimensions;

  for (int64_t reference_start = 0;
       reference_start < reference_points;
       reference_start += kWarpSize) {
    for (int position = k + lane;
         position < kSortSize;
         position += kWarpSize) {
      distances[position] = INFINITY;
      indices[position] = std::numeric_limits<int32_t>::max();
    }
    const int64_t reference_index = reference_start + lane;
    if (reference_index < reference_points) {
      distances[k + lane] = squared_distance(
          query_point,
          reference_batch + reference_index * dimensions,
          dimensions);
      indices[k + lane] = static_cast<int32_t>(reference_index);
    }
    __syncwarp();
    warp_bitonic_sort<kSortSize>(distances, indices, lane);
  }

  if (lane < k) {
    const int64_t output_position = query_linear * k + lane;
    output_distances[output_position] = distances[lane];
    output_indices[output_position] =
        static_cast<output_index_t>(indices[lane]);
  }
}

template <typename output_index_t>
__device__ __forceinline__ void swap_pair(
    float* distances,
    output_index_t* indices,
    const int lhs,
    const int rhs) {
  const float distance = distances[lhs];
  distances[lhs] = distances[rhs];
  distances[rhs] = distance;
  const output_index_t index = indices[lhs];
  indices[lhs] = indices[rhs];
  indices[rhs] = index;
}

template <typename output_index_t>
__device__ __forceinline__ void max_heap_sift_down(
    float* distances,
    output_index_t* indices,
    int root,
    const int heap_size) {
  while (true) {
    const int left = root * 2 + 1;
    if (left >= heap_size) {
      return;
    }
    const int right = left + 1;
    int larger = left;
    if (right < heap_size && pair_less(
            distances[left],
            static_cast<int32_t>(indices[left]),
            distances[right],
            static_cast<int32_t>(indices[right]))) {
      larger = right;
    }
    if (!pair_less(
            distances[root],
            static_cast<int32_t>(indices[root]),
            distances[larger],
            static_cast<int32_t>(indices[larger]))) {
      return;
    }
    swap_pair(distances, indices, root, larger);
    root = larger;
  }
}

// Generic K uses the output slice as an in-place max heap. Every distance is
// evaluated once and no workspace proportional to Nq*Nr is allocated.
template <typename scalar_t, typename output_index_t>
__global__ void heap_knn_kernel(
    float* __restrict__ output_distances,
    output_index_t* __restrict__ output_indices,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ reference,
    const int64_t query_points,
    const int64_t reference_points,
    const int64_t dimensions,
    const int k,
    const int64_t total_queries) {
  const int64_t query_linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query_linear >= total_queries) {
    return;
  }

  const int64_t batch = query_linear / query_points;
  const scalar_t* query_point = query + query_linear * dimensions;
  const scalar_t* reference_batch =
      reference + batch * reference_points * dimensions;
  float* distances = output_distances + query_linear * k;
  output_index_t* indices = output_indices + query_linear * k;

  for (int neighbor = 0; neighbor < k; ++neighbor) {
    distances[neighbor] = squared_distance(
        query_point,
        reference_batch + static_cast<int64_t>(neighbor) * dimensions,
        dimensions);
    indices[neighbor] = static_cast<output_index_t>(neighbor);
  }
  for (int root = k / 2 - 1; root >= 0; --root) {
    max_heap_sift_down(distances, indices, root, k);
  }

  for (int64_t reference_index = k;
       reference_index < reference_points;
       ++reference_index) {
    const float distance = squared_distance(
        query_point,
        reference_batch + reference_index * dimensions,
        dimensions);
    const int32_t index = static_cast<int32_t>(reference_index);
    if (pair_less(
            distance,
            index,
            distances[0],
            static_cast<int32_t>(indices[0]))) {
      distances[0] = distance;
      indices[0] = static_cast<output_index_t>(index);
      max_heap_sift_down(distances, indices, 0, k);
    }
  }

  for (int end = k - 1; end > 0; --end) {
    swap_pair(distances, indices, 0, end);
    max_heap_sift_down(distances, indices, 0, end);
  }
}

template <int K, typename scalar_t, typename output_index_t>
void launch_small_knn(
    torch::Tensor& output_distances,
    torch::Tensor& output_indices,
    const torch::Tensor& query,
    const torch::Tensor& reference) {
  constexpr int kSortSize = next_power_of_two(kWarpSize * K);
  const int64_t total_queries = query.size(0) * query.size(1);
  const int blocks = static_cast<int>(
      (total_queries + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const int threads = kWarpSize * kWarpsPerBlock;
  const size_t shared_bytes = static_cast<size_t>(
      kWarpsPerBlock * kSortSize * (sizeof(float) + sizeof(int32_t)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  small_knn_kernel<K, scalar_t, output_index_t>
      <<<blocks, threads, shared_bytes, stream>>>(
          output_distances.data_ptr<float>(),
          output_indices.data_ptr<output_index_t>(),
          query.data_ptr<scalar_t>(),
          reference.data_ptr<scalar_t>(),
          query.size(1),
          reference.size(1),
          query.size(2),
          total_queries);
}

template <typename scalar_t, typename output_index_t>
void launch_knn(
    torch::Tensor& output_distances,
    torch::Tensor& output_indices,
    const torch::Tensor& query,
    const torch::Tensor& reference) {
  const int k = static_cast<int>(output_indices.size(2));
  if (k <= 8) {
#define LAUNCH_SMALL(K_VALUE)                                               \
    case K_VALUE:                                                          \
      launch_small_knn<K_VALUE, scalar_t, output_index_t>(                 \
          output_distances, output_indices, query, reference);             \
      break
    switch (k) {
      LAUNCH_SMALL(1);
      LAUNCH_SMALL(2);
      LAUNCH_SMALL(3);
      LAUNCH_SMALL(4);
      LAUNCH_SMALL(5);
      LAUNCH_SMALL(6);
      LAUNCH_SMALL(7);
      LAUNCH_SMALL(8);
    }
#undef LAUNCH_SMALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  const int64_t total_queries = query.size(0) * query.size(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (k <= 32) {
    constexpr int kSortSize = 64;
    const int blocks = static_cast<int>(
        (total_queries + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const size_t shared_bytes = static_cast<size_t>(
        kWarpsPerBlock * kSortSize * (sizeof(float) + sizeof(int32_t)));
    warp_merge_knn_kernel<scalar_t, output_index_t>
        <<<blocks, kWarpSize * kWarpsPerBlock, shared_bytes, stream>>>(
            output_distances.data_ptr<float>(),
            output_indices.data_ptr<output_index_t>(),
            query.data_ptr<scalar_t>(),
            reference.data_ptr<scalar_t>(),
            query.size(1),
            reference.size(1),
            query.size(2),
            k,
            total_queries);
  } else {
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(
        (total_queries + kThreads - 1) / kThreads);
    heap_knn_kernel<scalar_t, output_index_t>
        <<<blocks, kThreads, 0, stream>>>(
            output_distances.data_ptr<float>(),
            output_indices.data_ptr<output_index_t>(),
            query.data_ptr<scalar_t>(),
            reference.data_ptr<scalar_t>(),
            query.size(1),
            reference.size(1),
            query.size(2),
            k,
            total_queries);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_index_type(
    torch::Tensor& output_distances,
    torch::Tensor& output_indices,
    const torch::Tensor& query,
    const torch::Tensor& reference) {
  if (output_indices.scalar_type() == torch::kInt32) {
    launch_knn<scalar_t, int32_t>(
        output_distances, output_indices, query, reference);
  } else {
    launch_knn<scalar_t, int64_t>(
        output_distances, output_indices, query, reference);
  }
}

}  // namespace

void knn_forward(
    torch::Tensor& output_distances,
    torch::Tensor& output_indices,
    const torch::Tensor& query,
    const torch::Tensor& reference) {
  TORCH_CHECK(
      query.is_cuda() && reference.is_cuda() && output_distances.is_cuda() &&
          output_indices.is_cuda(),
      "fused_knn expects CUDA tensors");
  TORCH_CHECK(
      query.is_contiguous() && reference.is_contiguous() &&
          output_distances.is_contiguous() && output_indices.is_contiguous(),
      "fused_knn expects contiguous tensors");
  TORCH_CHECK(
      query.device() == reference.device() &&
          query.device() == output_distances.device() &&
          query.device() == output_indices.device(),
      "fused_knn tensors must be on the same device");
  TORCH_CHECK(
      query.scalar_type() == reference.scalar_type(),
      "fused_knn query and reference must have the same dtype");
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kFloat32,
      "fused_knn supports float16 and float32 coordinates");
  TORCH_CHECK(
      output_distances.scalar_type() == torch::kFloat32,
      "fused_knn distances must be float32");
  TORCH_CHECK(
      output_indices.scalar_type() == torch::kInt32 ||
          output_indices.scalar_type() == torch::kInt64,
      "fused_knn indices must be int32 or int64");
  TORCH_CHECK(
      query.dim() == 3 && reference.dim() == 3,
      "fused_knn query/reference must have shape [B, N, D]");
  TORCH_CHECK(
      query.size(0) == reference.size(0) &&
          query.size(2) == reference.size(2),
      "fused_knn query/reference batch and coordinate dimensions must match");
  TORCH_CHECK(
      query.size(0) > 0 && query.size(1) > 0 && reference.size(1) > 0 &&
          query.size(2) > 0,
      "fused_knn query/reference dimensions must be non-zero");
  TORCH_CHECK(
      reference.size(1) <= std::numeric_limits<int32_t>::max(),
      "fused_knn reference point count must fit int32");

  TORCH_CHECK(
      output_indices.dim() == 3 && output_distances.dim() == 3,
      "fused_knn outputs must have shape [B, Nq, K]");
  const int64_t k = output_indices.size(2);
  TORCH_CHECK(
      k >= 1 && k <= kMaxK && k <= reference.size(1),
      "fused_knn requires 1 <= K <= min(1024, reference points)");
  TORCH_CHECK(
      output_distances.sizes() == output_indices.sizes() &&
          output_indices.size(0) == query.size(0) &&
          output_indices.size(1) == query.size(1),
      "fused_knn output tensors must have shape [B, Nq, K]");

  c10::cuda::CUDAGuard device_guard(query.device());
  if (query.scalar_type() == torch::kFloat16) {
    dispatch_index_type<at::Half>(
        output_distances, output_indices, query, reference);
  } else {
    dispatch_index_type<float>(
        output_distances, output_indices, query, reference);
  }
}

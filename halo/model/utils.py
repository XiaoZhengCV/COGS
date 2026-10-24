import torch
import torch.nn as nn
from torch.nn import SyncBatchNorm

sync_bn = SyncBatchNorm.convert_sync_batchnorm

def calc_pwd(query, reference=None):
	"""
	Compute pairwise squared Euclidean distances
		query:     Tensor of shape (B, C, Nq)
		reference: Tensor of shape (B, C, Nr). If None, compute distances between x and itself.
	Returns:
		dist: Tensor of shape (B, Nq, Nr) if y is provided, else (B, Nq, Nq)
	"""
	if reference is None:
		reference = query
	q_norm = torch.sum(query ** 2, dim=1, keepdim=True)  # (B, 1, Nq)
	r_norm = torch.sum(reference ** 2, dim=1, keepdim=True)  # (B, 1, Nr)
	return q_norm.transpose(1, 2) + r_norm + torch.bmm(query.transpose(1,2), reference.mul(-2))  # [B, N, M]


def pointops_knn(query, reference=None, k=1):
	"""CUDA kNN for channel-first point clouds.

	Args:
		query: [B, 3, Nq].
		reference: [B, 3, Nr]. If None, use query.
		k: number of neighbors.

	Returns:
		idx: [B, Nq, k], batch-local indices into reference.
		dist: [B, Nq, k], Euclidean distances.
	"""
	try:
		import pointops
	except ImportError as e:
		raise ImportError(
			"pointops is required for CUDA kNN; install it in the training environment"
		) from e

	if reference is None:
		reference = query
	if query.dim() != 3 or reference.dim() != 3:
		raise ValueError("query and reference must have shape [B, 3, N]")
	B, Cq, Nq = query.shape
	Br, Cr, Nr = reference.shape
	if B != Br or Cq != 3 or Cr != 3:
		raise ValueError("query and reference must have shape [B, 3, N]")
	if not 1 <= k <= Nr:
		raise ValueError(f"k must be in [1, {Nr}], got {k}")
	if not query.is_cuda or not reference.is_cuda:
		raise RuntimeError("pointops_knn requires CUDA tensors")

	# pointops uses flattened points and cumulative per-batch offsets.
	xyz = reference.float().transpose(1, 2).contiguous().view(B * Nr, 3)
	new_xyz = query.float().transpose(1, 2).contiguous().view(B * Nq, 3)
	offset = torch.arange(
		1, B + 1, device=xyz.device, dtype=torch.int32
	) * Nr
	new_offset = torch.arange(
		1, B + 1, device=new_xyz.device, dtype=torch.int32
	) * Nq

	idx, dist = pointops.knn_query(
		k, xyz, offset, new_xyz, new_offset
	)
	idx = idx.view(B, Nq, k).long()
	dist = dist.view(B, Nq, k)
	# pointops returns indices in the flattened [B * Nr] array.
	base = torch.arange(B, device=idx.device).view(B, 1, 1) * Nr
	return idx - base, dist


def knn(query, reference, k, transpose=False):
	"""
	Input:
		query:     Tensor of shape (B, C, Nq) | (B, Nq, C)
		reference: Tensor of shape (B, C, Nr) | (B, Nr, C)
		k:         int, number of nearest neighbors
		transpose: apply when query shape of (B, Nq, C)
	Output:
		idx:  Tensor of shape (B, Nq, k), indices of nearest neighbors in `reference`
		dist: Tensor of shape (B, Nq, k), distances to nearest neighbors
	"""
	# ||q - r||^2 = ||q||^2 + ||r||^2 - 2*q^T*r
	if transpose:
		query = query.transpose(2, 1)
		reference = reference.transpose(2, 1)
	dist = calc_pwd(query, reference)  # (B, Nq, Nr)
	dist_k, idx = dist.topk(k=k, dim=-1, largest=False, sorted=True)  # (B, Nq, k)
	return idx, dist_k


def index_points(points, idx):
	"""
	points: [B, C, N]
	idx: [B, S] or [B, N, K]
	returns: [B, C, S] or [B, C, N, K]
	"""
	_, C, _ = points.shape
	if idx.dim() == 2:  # [B, S]
		idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, S]
	elif idx.dim() == 3:  # [B, N, K]
		points = points.unsqueeze(-1).expand(-1, -1, -1, idx.shape[2])  # [B, C, N, K]
		idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)         # [B, C, N, K]
	return torch.gather(points, dim=2, index=idx_expanded) 


def get_graph_feature(x, k=20, idx=None, x_coord=None):
	"""Build DGCNN edge features using the bmm-based kNN above."""
	if idx is None:
		coord = x if x_coord is None else x_coord
		idx, _ = knn(coord, coord, k=k)
	feature = index_points(x, idx)
	x_expanded = x.unsqueeze(-1).expand(-1, -1, -1, k)
	return torch.cat([feature - x_expanded, x_expanded], dim=1)


class Conv(nn.Module):
	def __init__(self, in_channel, out_channel, dim=3, bias=False):
		super().__init__()
		Conv = nn.Conv2d if dim==4 else nn.Conv1d
		self.net = Conv(in_channel, out_channel, kernel_size=1, bias=bias)

	def forward(self, x):
		return self.net(x)


class ConvBN(nn.Module):
	def __init__(self, in_channel, out_channel, dim=3, bias=False):
		super().__init__()
		BatchNorm = nn.BatchNorm2d if dim==4 else nn.BatchNorm1d	
		self.net = nn.Sequential(
			Conv(in_channel, out_channel, dim, bias),
			BatchNorm(out_channel),
		)

	def forward(self, x):
		return self.net(x)


class ConvBNAct(nn.Module):
	def __init__(self, in_channel, out_channel, act=nn.ReLU(), dim=3, bias=False):
		super().__init__()
		self.net = ConvBN(in_channel, out_channel, dim, bias)
		self.act = act

	def forward(self, x):
		return self.act(self.net(x))
	

class LinearBNAct(nn.Module):
	def __init__(self, in_channel, out_channel, act=nn.ReLU(), bias=False):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_channel, out_channel, bias=bias),
			nn.BatchNorm1d(out_channel),
			act
		)

	def forward(self, x):
		return self.net(x)


class LinearBN(nn.Module):
	def __init__(self, in_channel, out_channel, bias=False):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_channel, out_channel, bias=bias),
			nn.BatchNorm1d(out_channel),
		)

	def forward(self, x):
		return self.net(x)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
	"""Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

	This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
	the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
	See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
	changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
	'survival rate' as the argument.

	"""
	if drop_prob == 0. or not training:
		return x
	keep_prob = 1 - drop_prob
	shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
	random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
	if keep_prob > 0.0 and scale_by_keep:
		random_tensor.div_(keep_prob)
	return x * random_tensor


class DropPath(nn.Module):
	"""Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
	"""
	def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob
		self.scale_by_keep = scale_by_keep

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

	def extra_repr(self):
		return f'drop_prob={round(self.drop_prob,3):0.3f}'

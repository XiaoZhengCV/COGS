import math
from typing import Union, List
from torch.optim import Optimizer
from mmengine.optim import BaseOptimWrapper
from mmengine.optim.scheduler import _ParamScheduler
from mmengine.registry import PARAM_SCHEDULERS

OptimizerType = Union[BaseOptimWrapper, Optimizer]
INF = int(1e9)

@PARAM_SCHEDULERS.register_module()
class TimmStyleCosineLR(_ParamScheduler):
    """
    继承自 PyTorch 原生 CosineAnnealingLR 以避免 MMEngine 接口兼容问题。
    
    功能：
    复现 timm 的 'warmup_prefix=False' 逻辑：
    1. 在 Warmup 阶段：线性增加。
    2. 在 Cosine 阶段：使用全局时间 t 计算余弦，而不重置时间。
       这意味着 Warmup 结束时，学习率不会处于最高点 (cos(0))，而是处于 cos(warmup/T_max) 的位置。
    """

    def __init__(self,
                 optimizer: OptimizerType,
                 t_initial: int,
                 param_name: str = 'lr',
                 lr_min: float = 0.,
                 cycle_mul: float = 1.,
                 cycle_decay: float = 1.,
                 cycle_limit: int = 1,
                 warmup_t: int = 0,
                 warmup_lr_init: float = 0,
                 warmup_prefix: bool = False,
                 k_decay: float = 1.0,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay

        super().__init__(
            optimizer=optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose
        )

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              t_initial,
                              warmup_t=0,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, but got {epoch_length}.'
        
        by_epoch = False
        t_initial = int(t_initial * epoch_length)
        warmup_t = int(warmup_t * epoch_length)
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
            
        return cls(
            *args,
            t_initial=t_initial,
            warmup_t=warmup_t,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self) -> List[float]:
        t = self.last_step

        if t < self.warmup_t:
            pct = t / self.warmup_t
            return [
                self.warmup_lr_init + pct * (v - self.warmup_lr_init) 
                for v in self.base_values
            ]

        if self.warmup_prefix:
            t = t - self.warmup_t

        if self.cycle_mul != 1:
            i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
            t_i = self.cycle_mul ** i * self.t_initial
            t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
        else:
            i = t // self.t_initial
            t_i = self.t_initial
            t_curr = t - (self.t_initial * i)

        gamma = self.cycle_decay ** i
        lr_max_values = [v * gamma for v in self.base_values]
        k = self.k_decay

        if i < self.cycle_limit:
            return [
                self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                for lr_max in lr_max_values
            ]
        else:
            return [self.lr_min for _ in self.base_values]
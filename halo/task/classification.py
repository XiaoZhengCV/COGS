import torch.nn as nn

from mmengine import MMLogger
from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class Classification(BaseModel):
	"""For classification task"""
	def __init__(self, backbone: dict, loss_fn: str='CrossEntropyLoss', label_smoothing: float=0.0):
		super().__init__()
		self.model = MODELS.build(backbone)
	
		total_params = sum(p.numel() for p in self.model.parameters())
		trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		logger = MMLogger.get_current_instance()
		logger.info(f"Total parameters: {total_params}, {total_params*4/(1024**2):.4f}MB")  
		logger.info(f"Trainable parameters: {trainable_params}, {trainable_params*4/(1024**2):.4f}MB") 

		if loss_fn=='CrossEntropyLoss':
			self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
		else:
			raise ValueError(f"loss_fn {loss_fn} not supported")
		self.train_gt = []
		self.train_pred = []

	def forward(self, mode = 'loss', **gs_attr):
		x = self.model(gs_attr)
		if not isinstance(x, tuple):
			x = (x, )

		if mode=='loss':
			self.train_gt.append(gs_attr['label'].cpu().numpy())
			self.train_pred.append(x[0].max(dim=1)[1].cpu().numpy())
			loss = self.loss_fn(x[0], gs_attr['label'])
			if len(x) > 1:
				loss += x[1]
			return {'loss': loss}
		elif mode=='predict':
			return x[0], gs_attr['label']
		elif mode=='tensor':
			return x[0]
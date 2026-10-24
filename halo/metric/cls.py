import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from mmengine import MMLogger
from mmengine.hooks import Hook
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS, HOOKS
from sklearn.metrics import accuracy_score, balanced_accuracy_score


@METRICS.register_module()
class SingleLabelMetrics(BaseMetric):
	"""
		Overall Accuracy (OA),
		Mean Class Accuracy (mAcc)
	"""
	default_prefix = 'Classification'
	def process(self, data_batch, data_samples):
		pred, gt = data_samples
		pred = pred.max(dim=1)[1].cpu().numpy()
		gt = gt.cpu().numpy()
		
		self.results.append({
			'predict': pred,
			'gt': gt
		})

	def compute_metrics(self, results):
		pred = []
		gt = []
		for item in results:
			pred.append(item['predict'])
			gt.append(item['gt'])
		pred = np.concatenate(pred)
		gt = np.concatenate(gt)
		return {
			'OverallAccuracy': accuracy_score(gt, pred),
			'MeanClassAccuracy': balanced_accuracy_score(gt, pred)
		}
	
@METRICS.register_module()
class MultiLabelMetrics(BaseMetric):
	"""
		Subset Accuracy
	"""
	default_prefix = 'Classification'
	def process(self, data_batch, data_samples):
		pred, gt = data_samples
		pred = (F.sigmoid(pred) > 0.5).cpu().numpy()
		gt = gt.cpu().numpy()
		
		self.results.append({
			'predict': pred,
			'gt': gt
		})

	def compute_metrics(self, results):
		pred = []
		gt = []
		for item in results:
			pred.append(item['predict'])
			gt.append(item['gt'])
		pred = np.concatenate(pred)
		gt = np.concatenate(gt)
		return {
			'subset_accuracy': accuracy_score(gt, pred),
		}
	
@HOOKS.register_module()
class TrainAccuracyLogHook(Hook):
	"""OA and mAcc for Training Set"""
	PRIORITY = 'NORMAL'  

	def after_train_epoch(self, runner):
		if hasattr(runner.model, 'module'):
			model = runner.model.module
			gt = self._gather_data(model.train_gt, runner).reshape(-1)
			pred = self._gather_data(model.train_pred, runner).reshape(-1)
		else:
			model = runner.model		
			gt = np.concatenate(model.train_gt)
			pred = np.concatenate(model.train_pred)

		acc = {
			'OverallAccuracy': accuracy_score(gt, pred),
			'MeanClassAccuracy': balanced_accuracy_score(gt, pred)
		}
		logger = MMLogger.get_current_instance()
		logger.info(f'Epoch(train)\t[{runner.epoch+1}]:\t{acc}')    
		
		model.train_gt = []
		model.train_pred = []

	def _gather_data(self, local_data, runner):
		world_size = runner.world_size
		gathered = [None] * world_size
		dist.all_gather_object(gathered, local_data)
		return np.concatenate(gathered)
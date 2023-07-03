import os
import shutil
import torch
import numpy as np
from torch.nn import functional as F

def save_model_by_name(model_dir, model, global_step, history=None):
	save_dir = os.path.join('checkpoints', model_dir, model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	if history is not None:
		np.save(os.path.join(save_dir, 'test_metrics_history'), history)
	print('Saved to {}'.format(file_path))

def load_model_by_name(model, global_step):
	"""
	Load a model based on its name model.name and the checkpoint iteration step

	Args:
		model: Model: (): A model
		global_step: int: (): Checkpoint iteration
	"""
	file_path = os.path.join('checkpoints', model_dir, train_mode, model.name,
							 'model.pt')
	state = torch.load(file_path)
	model.load_state_dict(state)
	print("Loaded from {}".format(file_path))

ce = torch.nn.CrossEntropyLoss(reduction='none')

def cross_entropy_loss(x, logits):
	"""
	Computes the log probability of a Bernoulli given its logits

	Args:
		x: tensor: (batch, dim): Observation
		logits: tensor: (batch, dim): Bernoulli logits

	Return:
		log_prob: tensor: (batch,): log probability of each sample
	"""
	log_prob = ce(input=logits, target=x).sum(-1)
	return log_prob

def kl_normal(qm, qv, pm, pv):
	"""
	Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
	sum over the last dimension

	Args:
		qm: tensor: (batch, dim): q mean
		qv: tensor: (batch, dim): q variance
		pm: tensor: (batch, dim): p mean
		pv: tensor: (batch, dim): p variance

	Return:
		kl: tensor: (batch,): kl between each sample
	"""
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl
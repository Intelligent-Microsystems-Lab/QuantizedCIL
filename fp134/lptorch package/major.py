import torch
import os
import lptorch_cuda

error_quant = None
activ_quant = None
weight_quant = None
grad_quant = None
master_quant = None

kerror_quant = None
kactiv_quant = None

scale_fluctuate = True
hysteresis_update = True

torch.manual_seed(0)

def get_error_quant():
	return error_quant

def set_error_quant(value):
	global error_quant
	from .quant import quant
	if type(value) is quant:
		error_quant = value
	else:
		error_quant = None
		print('type of quant must be quant.quant')

def get_activ_quant():
	return activ_quant

def set_activ_quant(value):
	global activ_quant
	from .quant import quant
	if type(value) is quant:
		activ_quant = value
	else:
		activ_quant = None
		print('type of quant must be quant.quant')

def get_weight_quant():
	return weight_quant

def set_weight_quant(value):
	global weight_quant
	from .quant import quant
	if type(value) is quant:
		weight_quant = value
	else:
		weight_quant = None
		print('type of quant must be quant.quant')

def get_grad_quant():
	return grad_quant

def set_grad_quant(value):
	global grad_quant
	from .quant import quant
	if type(value) is quant:
		grad_quant = value
	else:
		grad_quant = None
		print('type of quant must be quant.quant')

def get_master_quant():
	return master_quant

def set_master_quant(value):
	global master_quant
	from .quant import quant
	if type(value) is quant:
		master_quant = value
	else:
		master_quant = None
		print('type of quant must be quant.quant')

def get_kerror_quant():
	return kerror_quant

def set_kerror_quant(value):
	global kerror_quant
	from .quant import quant
	if type(value) is quant:
		kerror_quant = value
	else:
		kerror_quant = None
		print('type of quant must be quant.quant')

def get_kactiv_quant():
	return kactiv_quant

def set_kactiv_quant(value):
	global kactiv_quant
	from .quant import quant
	if type(value) is quant:
		kactiv_quant = value
	else:
		kactiv_quant = None
		print('type of quant must be quant.quant')

def get_scale_fluctuate():
	return scale_fluctuate

def set_scale_fluctuate(value):
	global scale_fluctuate
	if type(value) is bool:
		scale_fluctuate = value
	else:
		print('type of scale_fluctuate must be bool')

def get_hysteresis_update():
	return hysteresis_update

def set_hysteresis_update(value):
	global hysteresis_update
	if type(value) is bool:
		hysteresis_update = value
	else:
		print('type of hysteresis_update must be bool')

def linear_quantize(tensor, scale, bit_num, room, stochastic, ch_wise):
	if scale is None:
		if tensor.dim() is 4 and ch_wise:
			scale = tensor.abs().reshape(tensor.shape[0],-1).max(1)[0]
		else:
			scale = tensor.abs().max()
		scale = torch.where(scale>0, scale.log2().floor(), scale)
		scale = scale.int().add(room)
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	overflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	underflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	if stochastic:
		rand = torch.rand_like(tensor).contiguous()
		lptorch_cuda.linear_quantize_sr(cuda_id, tensor, bit_num, scale, overflow, underflow, room, rand)
	else:
		lptorch_cuda.linear_quantize(cuda_id, tensor, bit_num, scale, overflow, underflow, room)
	tensor = tensor.reshape(shape)
	tensor.scale = scale.clone()
	if tensor.dim() is 4 and ch_wise:
		overflow = overflow.reshape(shape[0],-1).max(1)[0].int()
		underflow = underflow.reshape(shape[0],-1).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	else:
		if overflow.max() == 1: scale.add_(1)
		elif underflow.max() == 0: scale.add_(-1)
	return tensor

def linear_hysteresis(pre_tensor, tensor, scale, bit_num, room, ch_wise):
	if scale is None:
		if tensor.dim() is 4 and ch_wise:
			scale = tensor.abs().reshape(tensor.shape[0],-1).max(1)[0]
		else:
			scale = tensor.abs().max()
		scale = torch.where(scale>0, scale.log2().floor(), scale)
		scale = scale.int().add(room)
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	pre_tensor = pre_tensor.reshape(-1)
	overflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	underflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	lptorch_cuda.linear_hysteresis(cuda_id, pre_tensor, tensor, bit_num, scale, overflow, underflow, room)
	tensor = tensor.reshape(shape)
	tensor.scale = scale.clone()
	if tensor.dim() is 4 and ch_wise:
		overflow = overflow.reshape(shape[0],-1).max(1)[0].int()
		underflow = underflow.reshape(shape[0],-1).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	else:
		if overflow.max() == 1: scale.add_(1)
		elif underflow.max() == 0: scale.add_(-1)
	return tensor

def custom_fp_quantize(tensor, scale, man, room, stochastic, ch_wise):
	if scale is None:
		if tensor.dim() is 4 and ch_wise:
			scale = tensor.abs().reshape(tensor.shape[0],-1).max(1)[0]
		else:
			scale = tensor.abs().max()
		scale = torch.where(scale>0, scale.log2().floor(), scale)
		scale = scale.int().add(room)
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	overflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	underflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	if stochastic:
		rand = torch.rand_like(tensor).contiguous()
		lptorch_cuda.custom_fp_quantize_sr(cuda_id, tensor, man, scale, overflow, underflow, room, rand)
	else:
		lptorch_cuda.custom_fp_quantize(cuda_id, tensor, man, scale, overflow, underflow, room)
	tensor = tensor.reshape(shape)
	tensor.scale = scale.clone()
	if tensor.dim() is 4 and ch_wise:
		overflow = overflow.reshape(shape[0],-1).max(1)[0].int()
		underflow = underflow.reshape(shape[0],-1).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	else:
		if overflow.max() == 1: scale.add_(1)
		elif underflow.max() == 0: scale.add_(-1)
	return tensor

def custom_fp_hysteresis(pre_tensor, tensor, scale, man, room, ch_wise):
	if scale is None:
		if tensor.dim() is 4 and ch_wise:
			scale = tensor.abs().reshape(tensor.shape[0],-1).max(1)[0]
		else:
			scale = tensor.abs().max()
		scale = torch.where(scale>0, scale.log2().floor(), scale)
		scale = scale.int().add(room)
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	pre_tensor = pre_tensor.reshape(-1)
	overflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	underflow = torch.empty(tensor.size(), dtype=torch.bool, device=tensor.device)
	lptorch_cuda.custom_fp_hysteresis(cuda_id, pre_tensor, tensor, man, scale, overflow, underflow, room)
	tensor = tensor.reshape(shape)
	tensor.scale = scale.clone()
	if tensor.dim() is 4 and ch_wise:
		overflow = overflow.reshape(shape[0],-1).max(1)[0].int()
		underflow = underflow.reshape(shape[0],-1).max(1)[0].int()
		overflow = torch.where(overflow>0, overflow, overflow.mul(0).add(underflow-1))
		scale.add_(overflow)
	else:
		if overflow.max() == 1: scale.add_(1)
		elif underflow.max() == 0: scale.add_(-1)
	return tensor

def fp_quantize(tensor, exp_bit, man_bit, bias, stochastic):
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	if stochastic:
		rand = torch.rand_like(tensor).contiguous()
		lptorch_cuda.fp_quantize_sr(cuda_id, tensor, exp_bit, man_bit, bias, rand)
	else:
		lptorch_cuda.fp_quantize(cuda_id, tensor, exp_bit, man_bit, bias)
	tensor = tensor.reshape(shape)
	tensor.scale = torch.zeros(1).int().to(tensor.device)
	return tensor

def fp_hysteresis(pre_tensor, tensor, exp_bit, man_bit, bias):
	cuda_id = tensor.get_device()
	shape = tensor.shape
	tensor = tensor.reshape(-1)
	pre_tensor = pre_tensor.reshape(-1)
	lptorch_cuda.fp_hysteresis(cuda_id, pre_tensor, tensor, exp_bit, man_bit, bias)
	tensor = tensor.reshape(shape)
	tensor.scale = torch.zeros(1).int().to(tensor.device)
	return tensor

def load_state_dict(target_model, state):
	t_state = target_model.state_dict()
	
	saved_key = state.keys()
	target_key = t_state.keys()
	missing_key = list(set(target_key) - set(saved_key))
	unexpected_key = list(set(saved_key) - set(target_key))
	
	missing_wo = []
	unexpected_wo = []
	for i in range(len(missing_key)):
		missing = missing_key[i].replace('module.', '')
		if 'cb_conv' in missing:
			missing_wo.append(missing.replace('cb_conv.', ''))
		elif 'cb_bn' in missing:
			sp = missing.split('.')
			for idx, text in enumerate(sp):
				if text == 'cb_bn':
					if idx != 0 and sp[idx-1].isdigit():
						sp[idx-1] = str(int(sp[idx-1])+1)
						del sp[idx]
					break
			key = sp.pop(0)
			for text in sp:
				key += '.'+text
			missing_wo.append(key)
		else:
		    missing_wo.append(missing)
	for i in range(len(unexpected_key)):
		unexpected = unexpected_key[i].replace('module.', '')
		if 'cb_conv' in unexpected:
			unexpected_wo.append(unexpected.replace('cb_conv.', ''))
		elif 'cb_bn' in unexpected:
			sp = unexpected.split('.')
			for idx, text in enumerate(sp):
				if text == 'cb_bn':
					if idx != 0 and sp[idx-1].isdigit():
						sp[idx-1] = str(int(sp[idx-1])+1)
						del sp[idx]
					break
			key = sp.pop(0)
			for text in sp:
				key += '.'+text
			unexpected_wo.append(key)
		else:
			unexpected_wo.append(unexpected)
	for j in range(len(unexpected_wo)):
		for i in range(len(missing_wo)):
			if unexpected_wo[j] == missing_wo[i]:
				state[missing_key[i]] = state.pop(unexpected_key[j])
	
	state = {k: v for k, v in state.items() if k in t_state}
	t_state.update(state)
	target_model.load_state_dict(t_state)
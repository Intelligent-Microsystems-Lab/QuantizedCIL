import torch
from . import functions as F
from torch.nn.utils.rnn import PackedSequence

class qblock_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale_fluct, scale, dual, training, tracking, bn, quantize):
        ctx.scale = scale
        ctx.dual = dual
        ctx.tracking = tracking
        ctx.quantize = quantize
        from .major import activ_quant, scale_fluctuate
        if activ_quant is None or quantize[0] is False:
            return input, scale_fluct
        if scale_fluctuate and training:
            if bn: scale_fluct = scale_fluct.mul(0)
            scale[0].add_(scale_fluct.int().item())
        if not training:
            scale[0] = scale[0].clone()
        forward_scale = scale[0].clone()
        if tracking[0]:
            if dual[0]:
                output = activ_quant.quantize(input.clone(), scale[0])
                output.add_(activ_quant.quantize(input.add(-output), forward_scale.add(-activ_quant.qformat.scale_diff)))
            else:
                output = activ_quant.quantize(input, scale[0])
            scale_fluct = scale_fluct.add(scale[0].add(-output.scale).float())
        else:
            if dual[0]:
                output = activ_quant.quantize(input.clone())
                output.add_(activ_quant.quantize(input.add(-output)))
            else:
                output = activ_quant.quantize(input)
            scale_fluct = scale_fluct.add(output.scale.add(-scale[0]).float())
            scale[0].data = output.scale.clone().data
        return output, scale_fluct

    @staticmethod
    def backward(ctx, grad_output, scale_fluct):
        scale = ctx.scale
        dual = ctx.dual
        tracking = ctx.tracking
        quantize = ctx.quantize
        from .major import error_quant, scale_fluctuate
        if error_quant is None or quantize[1] is False:
            return grad_output, scale_fluct, None, None, None, None, None, None
        if scale_fluctuate:
            if scale_fluct is None:
                scale_fluct = torch.tensor([0]).to(grad_output.device)
            scale[1].add_(scale_fluct.int().item())
        backward_scale = scale[1].clone()
        if tracking[1]:
            if dual[1]:
                grad_input = error_quant.quantize(grad_output.clone(), scale[1])
                grad_input.add_(error_quant.quantize(grad_output.add(-grad_input), backward_scale.add(-error_quant.qformat.scale_diff)))
            else:
                grad_input = error_quant.quantize(grad_output, scale[1])
            scale_fluct = scale_fluct.add(scale[1].add(-grad_input.scale).float())
        else:
            if dual[1]:
                grad_input = error_quant.quantize(grad_output.clone())
                grad_input.add_(error_quant.quantize(grad_output.add(-grad_input)))
            else:
                grad_input = error_quant.quantize(grad_output)
            scale_fluct = scale_fluct.add(grad_input.scale.add(-scale[1]).float())
            scale[1].data = grad_input.scale.clone().data
        return grad_input, scale_fluct, None, None, None, None, None, None

class QBlock(torch.nn.Module):
    def __init__(self, dual, fixed_scale, tracking, bn, quantize):
        super().__init__()
        self.dual = dual
        self.fixed_scale = fixed_scale
        self.tracking = tracking
        self.bn = bn
        self.quantize = quantize
        for i in range(2):
            if fixed_scale[i] is None:
                self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))
            else:
                self.register_buffer('scale'+str(i+1), torch.tensor([fixed_scale[i]], dtype=torch.int))

    def forward(self, inputs):
        scale = []
        for i in range(2):
            if self.fixed_scale[i] is None and self.training:
                scale.append(getattr(self, 'scale'+str(i+1)))
            else:
                scale.append(getattr(self, 'scale'+str(i+1)).clone())
        return qblock_func.apply(inputs[0], inputs[1], scale, self.dual, self.training, self.tracking, self.bn, self.quantize)

class QLayer(torch.nn.Module):
    def __init__(self, module=None, function=None, dual=[False, False], fixed_scale=[None, None], last=False, tracking=[True, True], quantize=[True, True], ret_dict = False):
        super().__init__()
        self.module = module
        self.function = function
        self.last = last
        self.ret_dict = ret_dict
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn = True
        else:
            bn = False
        self.qblock = QBlock(dual, fixed_scale, tracking, bn, quantize)

    def forward(self, inputs):
        if type(inputs) is not tuple:
            a, b = inputs, torch.zeros(1).to(inputs.device)
        else:
            a, b = inputs
        if self.module is not None:
            a = self.module(a)
        if self.function is not None:
            a = self.function(a)
        outputs = self.qblock((a,b))
        if self.last:
            if self.ret_dict:
                return {'logits': outputs[0]}
            return outputs[0]
        else:
            return outputs

class NQLayer(torch.nn.Module):
    def __init__(self, module=None, function=None, last=False):
        super().__init__()
        self.module = module
        self.function = function
        self.last = last
    
    def forward(self, inputs):
        if type(inputs) is not tuple:
            a, b = inputs, torch.zeros(1).to(inputs.device)
        else:
            a, b = inputs
        if hasattr(a, 'scale'): scale = a.scale
        else: scale = None
        if self.module is not None:
            a = self.module(a)
        if self.function is not None:
            a = self.function(a)
        a.scale = scale
        outputs = a, b
        if self.last:
            return outputs[0]
        else:
            return outputs 

class qadd_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, scale_fluct1, b, scale_fluct2, scale, training):
        from .major import activ_quant, scale_fluctuate
        if activ_quant is None:
            return a.add(b), torch.zeros(1).to(a.device)
        scale_fluct = torch.max(a.scale.float(), b.scale.float()).add(-torch.max(a.scale.add(-scale_fluct1), b.scale.add(-scale_fluct2)).float())
        if scale_fluctuate and training:
            scale[0].add_(scale_fluct.int())
        if not training:
            scale[0] = scale[0].clone()
        output = activ_quant.quantize(a.add(b), scale[0])
        # output = activ_quant.quantize(a.add(b))
        scale_fluct = scale_fluct.add(scale[0].add(-output.scale).float())
        return output, scale_fluct

    @staticmethod
    def backward(ctx, grad_output, scale_fluct):
        return grad_output, scale_fluct, grad_output, scale_fluct, None, None

class QAdd(torch.nn.Module):
    def __init__(self, last=False):
        super().__init__()
        self.last = last
        self.register_buffer('scale1', torch.tensor([0], dtype=torch.int))

    def forward(self, a, b):
        if type(a) is not tuple:
            a = [a, torch.zeros(1).to(a.device)]
        if type(b) is not tuple:
            b = [b, torch.zeros(1).to(b.device)]
        scale = [self.scale1]
        a, b = qadd_func.apply(a[0], a[1], b[0], b[1], scale, self.training)
        if self.last:
            return a
        return a, b

class qclone_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, scale_fluct, scale):
        b = a.clone().detach()
        if hasattr(a, 'scale') and a.scale is not None:
            b.scale = a.scale.clone().detach()
        scale_fluct2 = scale_fluct.clone().detach()
        ctx.scale = scale
        return a, scale_fluct, b, scale_fluct2

    @staticmethod
    def backward(ctx, grad_output1, scale_fluct1, grad_output2, scale_fluct2):
        scale = ctx.scale
        from .major import error_quant, scale_fluctuate
        if error_quant is None:
            return grad_output1.add(grad_output2), torch.zeros(1).to(grad_output1.device), None
        if scale_fluctuate:
            scale[0].add_(scale_fluct1.int())
            scale[1].add_(scale_fluct2.int())
        grad_output1 = error_quant.quantize(grad_output1, scale[0])
        grad_output2 = error_quant.quantize(grad_output2, scale[1])
        
        grad_input = grad_output1.add(grad_output2)
        scale_fluct1 = scale_fluct1.add(scale[0].add(-grad_output1.scale).float())
        scale_fluct2 = scale_fluct2.add(scale[1].add(-grad_output2.scale).float())

        scale_fluct = torch.max(grad_output1.scale.float(), grad_output2.scale.float()).add(-torch.max(grad_output1.scale.add(-scale_fluct1), grad_output2.scale.add(-scale_fluct2)).float())
        return grad_input, scale_fluct, None

class QClone(torch.nn.Module):
    def __init__(self, last=False):
        super().__init__()
        self.last = last
        for i in range(2):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, inputs):
        if type(inputs) is not tuple:
            a, b = inputs, torch.zeros(1).to(inputs.device)
        else:
            a, b = inputs
        scale = []
        for i in range(2):
            scale.append(getattr(self, 'scale'+str(i+1)))
        a1,a2,b1,b2 = qclone_func.apply(a, b, scale)
        if self.last:
            return a1, b1
        a = (a1,a2)
        b = (b1,b2)
        return a, b
    
class lstm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, hx, bias, num_layers, dropout, training, bidirectional, batch_first, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        h = hx[0].clone()
        c = hx[1].clone()
        hidden_size = h.shape[2]
        if batch_first:
            input = input.permute(1,0,2)
        seq_len = input.shape[0]
        for l in range(num_layers):
            c_scale_max = scale[l*5].add(-2).clone()
            if l != 0:
                input = torch.stack(h_array)
                if dropout > 0 and training:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand/(1-dropout))
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    c_scale = scale[l*5].clone()
                    if bias:
                        flat = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2]).add(h[num_directions*l+j].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3]))
                    else:
                        flat = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1)).add(h[num_directions*l+j].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1)))
                    i = flat[:,0*hidden_size:1*hidden_size]
                    f = flat[:,1*hidden_size:2*hidden_size]
                    g = flat[:,2*hidden_size:3*hidden_size]
                    o = flat[:,3*hidden_size:4*hidden_size]

                    if activ_quant is None:
                        ii = torch.sigmoid(i)
                        ff = torch.sigmoid(f)
                        gg = torch.tanh(g)
                        oo = torch.sigmoid(o)
                        gate = torch.stack((ii,ff,gg,oo,h[num_directions*l+j],c[num_directions*l+j])).clone()
                        c[num_directions*l+j] = c[num_directions*l+j] * ff + ii * gg
                        tanhc = torch.tanh(c[num_directions*l+j])
                        h[num_directions*l+j] = oo * tanhc
                    else:
                        ii = activ_quant.quantize(torch.sigmoid(i), zero.clone())
                        ff = activ_quant.quantize(torch.sigmoid(f), zero.clone())
                        gg = activ_quant.quantize(torch.tanh(g), zero.clone())
                        oo = activ_quant.quantize(torch.sigmoid(o), zero.clone())
                        gate = torch.stack((ii,ff,gg,oo,h[num_directions*l+j],c[num_directions*l+j])).clone()
                        c[num_directions*l+j] = activ_quant.quantize(c[num_directions*l+j] * ff + ii * gg, c_scale)
                        tanhc = activ_quant.quantize(torch.tanh(c[num_directions*l+j]), zero.clone())
                        h[num_directions*l+j] = activ_quant.quantize(oo * tanhc, zero.clone())
                        c_scale_max = torch.max(c_scale_max, c_scale)
                    
                    gate_array.append(torch.cat((gate, torch.unsqueeze(tanhc, dim=0))).clone())

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j].clone())
            scale[l*5].data = c_scale_max.data
        output = torch.stack(h_array)
        if batch_first:
            output = output.permute(1,0,2)
        
        if training:
            gate_tensor = torch.stack(gate_array)
            if num_layers == 1:
                ctx.save_for_backward(gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.batch_first = batch_first
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h, c

    @staticmethod
    def backward(ctx, grad_output, grad_hidden, grad_cell):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        batch_first = ctx.batch_first
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[2]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        if batch_first:
            grad_output = grad_output.permute(1,0,2)
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            gc_scale_max = scale[l*5+1].add(-2).clone()
            ggate_scale_max = scale[l*5+2].add(-2).clone()
            gh_scale_max = scale[l*5+3].add(-2).clone()
            gx_scale_max = scale[l*5+4].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l]/(1-dropout))
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    gc_scale = scale[l*5+1].clone()
                    ggate_scale = scale[l*5+2].clone()
                    gh_scale = scale[l*5+3].clone()
                    gx_scale = scale[l*5+4].clone()

                    gate = gate_tensor[(l*num_directions+j)*seq_len+k]
                    ii, ff, gg, oo, h, c, cc = gate[0], gate[1], gate[2], gate[3], gate[4], gate[5], gate[6]
                    alpha = grad_hidden[num_directions*l+j].add(grad_output[(1-2*j)*k-j,:,hidden_size*j:hidden_size*(j+1)]) * oo * (1 - cc * cc) + grad_cell[num_directions*l+j]
                    gi = alpha * gg * ii * (1 - ii)
                    gf = alpha * c * ff * (1 - ff)
                    gg = alpha * ii * (1 - gg * gg)
                    go = grad_hidden[num_directions*l+j].add(grad_output[(1-2*j)*k-j,:,hidden_size*j:hidden_size*(j+1)]) * cc * oo * (1 - oo)
                    
                    if error_quant is None:
                        grad_cell[num_directions*l+j] = alpha * ff
                        g_gate = torch.cat((gi,gf,gg,go),dim=1)
                        grad_hidden[num_directions*l+j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1])
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                    else:
                        grad_cell[num_directions*l+j] = error_quant.quantize(alpha * ff, gc_scale)
                        g_gate = error_quant.quantize(torch.cat((gi,gf,gg,go),dim=1), ggate_scale)
                        grad_hidden[num_directions*l+j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]), gh_scale)
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[(1-2*j)*k-j] = error_quant.quantize(grad_input[(1-2*j)*k-j], gx_scale)
                        gc_scale_max = torch.max(gc_scale_max, gc_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)

                    grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[(1-2*j)*k-j]))
                    grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                    if bias:
                        grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
            scale[l*5+1].data = gc_scale_max.data
            scale[l*5+2].data = ggate_scale_max.data
            scale[l*5+3].data = gh_scale_max.data
            scale[l*5+4].data = gx_scale_max.data
        
        if batch_first:
            grad_input = grad_input.permute(1,0,2)
        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class packed_lstm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, batch_sizes, hx, bias, num_layers, dropout, training, bidirectional, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        gate_cat_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        h = hx[0].clone()
        c = hx[1].clone()
        hidden_size = h.shape[2]
        seq_len = batch_sizes.shape[0]
        for l in range(num_layers):
            c_scale_max = scale[l*5].add(-2).clone()
            if l != 0:
                input = torch.cat(h_array)
                if dropout > 0 and training:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand/(1-dropout))
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    c_scale = scale[l*5].clone()
                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if bias:
                        flat = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2]).add(h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3]))
                    else:
                        flat = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1)).add(h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1)))
                    i = flat[:,0*hidden_size:1*hidden_size]
                    f = flat[:,1*hidden_size:2*hidden_size]
                    g = flat[:,2*hidden_size:3*hidden_size]
                    o = flat[:,3*hidden_size:4*hidden_size]
                    
                    if activ_quant is None:
                        ii = torch.sigmoid(i)
                        ff = torch.sigmoid(f)
                        gg = torch.tanh(g)
                        oo = torch.sigmoid(o)
                        gate = torch.stack((ii,ff,gg,oo,h[num_directions*l+j][:current_batch_size],c[num_directions*l+j][:current_batch_size])).clone()
                        c[num_directions*l+j][:current_batch_size] = c[num_directions*l+j][:current_batch_size] * ff + ii * gg
                        tanhc = torch.tanh(c[num_directions*l+j][:current_batch_size])
                        h[num_directions*l+j][:current_batch_size] = oo * tanhc
                    else:
                        ii = activ_quant.quantize(torch.sigmoid(i), zero.clone())
                        ff = activ_quant.quantize(torch.sigmoid(f), zero.clone())
                        gg = activ_quant.quantize(torch.tanh(g), zero.clone())
                        oo = activ_quant.quantize(torch.sigmoid(o), zero.clone())
                        gate = torch.stack((ii,ff,gg,oo,h[num_directions*l+j][:current_batch_size],c[num_directions*l+j][:current_batch_size])).clone()
                        c[num_directions*l+j][:current_batch_size] = activ_quant.quantize(c[num_directions*l+j][:current_batch_size] * ff + ii * gg, c_scale)
                        tanhc = activ_quant.quantize(torch.tanh(c[num_directions*l+j][:current_batch_size]), zero.clone())
                        h[num_directions*l+j][:current_batch_size] = activ_quant.quantize(oo * tanhc, zero.clone())
                        c_scale_max = torch.max(c_scale_max, c_scale)

                    gate_array.append(torch.cat((gate, torch.unsqueeze(tanhc, dim=0))).clone())

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j][:current_batch_size].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j][:current_batch_size].clone())
                gate_cat_array.append(torch.cat(gate_array, dim=1))
                gate_array = []
            scale[l*5].data = c_scale_max.data
        output = torch.cat(h_array)
        
        if training:
            gate_tensor = torch.stack(gate_cat_array)
            if num_layers == 1:
                ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h, c

    @staticmethod
    def backward(ctx, grad_output, grad_hidden, grad_cell):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            batch_sizes, gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            batch_sizes, gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            batch_sizes, gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[1]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            gc_scale_max = scale[l*5+1].add(-2).clone()
            ggate_scale_max = scale[l*5+2].add(-2).clone()
            gh_scale_max = scale[l*5+3].add(-2).clone()
            gx_scale_max = scale[l*5+4].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l]/(1-dropout))
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    gc_scale = scale[l*5+1].clone()
                    ggate_scale = scale[l*5+2].clone()
                    gh_scale = scale[l*5+3].clone()
                    gx_scale = scale[l*5+4].clone()

                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if j is 1:
                        gate = gate_tensor[l*num_directions+j,:,batch_sizes.sum()-start_idx-current_batch_size:batch_sizes.sum()-start_idx,:]
                    else:
                        gate = gate_tensor[l*num_directions+j,:,start_idx:start_idx+current_batch_size,:]
                    ii, ff, gg, oo, h, c, cc = gate[0], gate[1], gate[2], gate[3], gate[4], gate[5], gate[6]
                    alpha = grad_hidden[num_directions*l+j][:current_batch_size].add(grad_output[start_idx:start_idx+current_batch_size,hidden_size*j:hidden_size*(j+1)]) * oo * (1 - cc * cc) + grad_cell[num_directions*l+j][:current_batch_size]
                    gi = alpha * gg * ii * (1 - ii)
                    gf = alpha * c * ff * (1 - ff)
                    gg = alpha * ii * (1 - gg * gg)
                    go = grad_hidden[num_directions*l+j][:current_batch_size].add(grad_output[start_idx:start_idx+current_batch_size,hidden_size*j:hidden_size*(j+1)]) * cc * oo * (1 - oo)
                    
                    if error_quant is None:
                        grad_cell[num_directions*l+j][:current_batch_size] = alpha * ff
                        g_gate = torch.cat((gi,gf,gg,go),dim=1)
                        grad_hidden[num_directions*l+j][:current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1])
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                    else:
                        grad_cell[num_directions*l+j][:current_batch_size] = error_quant.quantize(alpha * ff, gc_scale)
                        g_gate = error_quant.quantize(torch.cat((gi,gf,gg,go),dim=1), ggate_scale)
                        grad_hidden[num_directions*l+j][:current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]), gh_scale)
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_input[start_idx:start_idx+current_batch_size], gx_scale)
                        gc_scale_max = torch.max(gc_scale_max, gc_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)

                    grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[start_idx:start_idx+current_batch_size]))
                    grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                    if bias:
                        grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
            scale[l*5+1].data = gc_scale_max.data
            scale[l*5+2].data = ggate_scale_max.data
            scale[l*5+3].data = gh_scale_max.data
            scale[l*5+4].data = gx_scale_max.data
            
        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class LSTM(torch.nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, batch_first = False, dropout = 0., bidirectional = False):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        for i in range(num_layers*5):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, input, hx=None):  # noqa: F811
        if type(input) is tuple:
            input, _ = input
        scale = []
        for i in range(self.num_layers*5):
            scale.append(getattr(self, 'scale'+str(i+1)))

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = lstm_func.apply(input, scale, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first, *tuple(self._flat_weights))
        else:
            result = packed_lstm_func.apply(input, scale, batch_sizes, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, *tuple(self._flat_weights))
        
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)

class gru_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, hx, bias, num_layers, dropout, training, bidirectional, batch_first, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        hidden_size = hx.shape[2]
        h = hx.clone()
        if batch_first:
            input = input.permute(1,0,2)
        seq_len = input.shape[0]
        for l in range(num_layers):
            h_scale_max = scale[l*4].add(-2).clone()
            if l != 0:
                input = torch.stack(h_array)
                if dropout > 0 and training:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand/(1-dropout))
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    h_scale = scale[l*4].clone()
                    if bias:
                        flat_x = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2])
                        flat_h = h[num_directions*l+j].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3])
                    else:
                        flat_x = input[(1-2*j)*k-j,:,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1))
                        flat_h = h[num_directions*l+j].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1))
                    r = flat_x[:,0*hidden_size:1*hidden_size].add(flat_h[:,0*hidden_size:1*hidden_size])
                    z = flat_x[:,1*hidden_size:2*hidden_size].add(flat_h[:,1*hidden_size:2*hidden_size])
                    
                    if activ_quant is None:
                        rr = torch.sigmoid(r)
                        zz = torch.sigmoid(z)
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = torch.tanh(n)
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j].mul_(zz).add_((1 - zz) * nn)
                    else:
                        rr = activ_quant.quantize(torch.sigmoid(r), zero.clone())
                        zz = activ_quant.quantize(torch.sigmoid(z), zero.clone())
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = activ_quant.quantize(torch.tanh(n), zero.clone())
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j] = activ_quant.quantize(h[num_directions*l+j].mul(zz).add((1 - zz) * nn), h_scale)
                        h_scale_max = torch.max(h_scale_max, h_scale)

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j].clone())
            scale[l*4].data = h_scale_max.data
        output = torch.stack(h_array)
        if batch_first:
            output = output.permute(1,0,2)
        
        if training:
            gate_tensor = torch.stack(gate_array)
            if num_layers == 1:
                ctx.save_for_backward(gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.batch_first = batch_first
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h
    
    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        batch_first = ctx.batch_first
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[2]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        if batch_first:
            grad_output = grad_output.permute(1,0,2)
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            ggate_scale_max = scale[l*4+1].add(-2).clone()
            gx_scale_max = scale[l*4+2].add(-2).clone()
            gh_scale_max = scale[l*4+3].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l]/(1-dropout))
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    ggate_scale = scale[l*4+1].clone()
                    gx_scale = scale[l*4+2].clone()
                    gh_scale = scale[l*4+3].clone()
                    gate = gate_tensor[(l*num_directions+j)*seq_len+k]
                    rr, zz, nn, h, flat_h = gate[0], gate[1], gate[2], gate[3], gate[4]
                    alpha = grad_hidden[num_directions*l+j].add(grad_output[(1-2*j)*k-j,:,hidden_size*j:hidden_size*(j+1)])
                    gn = alpha * (1 - zz) * (1 - nn * nn)
                    gz = alpha * (h - nn) * zz * (1 - zz)
                    gr = gn * flat_h * rr * (1 - rr)

                    if error_quant is None:
                        g_gate = torch.cat((gr,gz,gn),dim=1)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[(1-2*j)*k-j]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[(1-2*j)*k-j].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:   
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        g_gate[:,2*hidden_size:].mul_(rr)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz
                    else:
                        g_gate = error_quant.quantize(torch.cat((gr,gz,gn),dim=1), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[(1-2*j)*k-j]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[(1-2*j)*k-j] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(grad_output_temp[(1-2*j)*k-j].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[(1-2*j)*k-j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:   
                            grad_input[(1-2*j)*k-j].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[(1-2*j)*k-j] = error_quant.quantize(grad_input[(1-2*j)*k-j], gx_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        ggate_scale = scale[l*4+1].clone()
                        g_gate[:,2*hidden_size:] = error_quant.quantize(g_gate[:,2*hidden_size:].mul(rr), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz, gh_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
            scale[l*4+1].data = ggate_scale_max.data
            scale[l*4+2].data = gx_scale_max.data
            scale[l*4+3].data = gh_scale_max.data
        
        if batch_first:
            grad_input = grad_input.permute(1,0,2)
        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class packed_gru_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, batch_sizes, hx, bias, num_layers, dropout, training, bidirectional, *_flat_weights):
        from .major import activ_quant
        zero = torch.ones(1).int().to(input.device)
        input_array = []
        gate_array = []
        gate_cat_array = []
        dropout_array = []
        
        num_directions = 2 if bidirectional else 1
        hidden_size = hx.shape[2]
        h = hx.clone()
        seq_len = batch_sizes.shape[0]
        for l in range(num_layers):
            h_scale_max = scale[l*4].add(-2).clone()
            if l != 0:
                input = torch.cat(h_array)
                if dropout > 0 and training:
                    rand = torch.rand_like(input)
                    true_tensor = torch.tensor(1, dtype=torch.bool).to(rand.device)
                    false_tensor = torch.tensor(0, dtype=torch.bool).to(rand.device)
                    rand = torch.where(rand > dropout, true_tensor, false_tensor)
                    input.mul_(rand/(1-dropout))
                    dropout_array.append(rand)
            h_array = []
            input_array.append(input.clone())
            for j in range(num_directions):
                for k in range(seq_len):
                    h_scale = scale[l*4].clone()
                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if bias:
                        flat_x = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[4*num_directions*l+4*j+0].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+2])
                        flat_h = h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[4*num_directions*l+4*j+1].transpose(0,1)).add(_flat_weights[4*num_directions*l+4*j+3])
                    else:
                        flat_x = input[start_idx:start_idx+current_batch_size,:].matmul(_flat_weights[2*num_directions*l+2*j+0].transpose(0,1))
                        flat_h = h[num_directions*l+j][:current_batch_size].matmul(_flat_weights[2*num_directions*l+2*j+1].transpose(0,1))
                    r = flat_x[:,0*hidden_size:1*hidden_size].add(flat_h[:,0*hidden_size:1*hidden_size])
                    z = flat_x[:,1*hidden_size:2*hidden_size].add(flat_h[:,1*hidden_size:2*hidden_size])
                    
                    if activ_quant is None:
                        rr = torch.sigmoid(r)
                        zz = torch.sigmoid(z)
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = torch.tanh(n)
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j][:current_batch_size],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j][:current_batch_size].mul_(zz).add_((1 - zz) * nn)
                    else:
                        rr = activ_quant.quantize(torch.sigmoid(r), zero.clone())
                        zz = activ_quant.quantize(torch.sigmoid(z), zero.clone())
                        n = flat_x[:,2*hidden_size:3*hidden_size].add(rr * flat_h[:,2*hidden_size:3*hidden_size])                
                        nn = activ_quant.quantize(torch.tanh(n), zero.clone())
                        gate_array.append(torch.stack((rr,zz,nn,h[num_directions*l+j][:current_batch_size],flat_h[:,2*hidden_size:3*hidden_size])).clone())
                        h[num_directions*l+j][:current_batch_size] = activ_quant.quantize(h[num_directions*l+j][:current_batch_size].mul(zz).add((1 - zz) * nn), h_scale)
                        h_scale_max = torch.max(h_scale_max, h_scale)

                    if j is 1:
                        h_array[seq_len-1-k] = torch.cat((h_array[seq_len-1-k], h[num_directions*l+j][:current_batch_size].clone()), dim=1)
                    else:
                        h_array.append(h[num_directions*l+j][:current_batch_size].clone())
                gate_cat_array.append(torch.cat(gate_array, dim=1))
                gate_array = []
            scale[l*4].data = h_scale_max.data
        output = torch.cat(h_array)
        
        if training:
            gate_tensor = torch.stack(gate_cat_array)
            if num_layers == 1:
                ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], *_flat_weights)
            else:
                input_tensor = torch.stack(input_array[1:])
                if dropout > 0: 
                    dropout_tensor = torch.stack(dropout_array)
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, dropout_tensor, *_flat_weights)
                else:
                    ctx.save_for_backward(batch_sizes, gate_tensor, input_array[0], input_tensor, *_flat_weights)
            ctx.bias = bias
            ctx.num_layers = num_layers
            ctx.dropout = dropout
            ctx.num_directions = num_directions
            ctx.seq_len = seq_len
            ctx.scale = scale
        
        return output, h

    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        from .major import error_quant
        grad_output = grad_output.clone()
        bias = ctx.bias
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        num_directions = ctx.num_directions
        seq_len = ctx.seq_len
        scale = ctx.scale
        if num_layers == 1:
            batch_sizes, gate_tensor, first_input, *_flat_weights = ctx.saved_tensors
        elif dropout > 0:
            batch_sizes, gate_tensor, first_input, input_tensor, dropout_tensor, *_flat_weights = ctx.saved_tensors
        else:
            batch_sizes, gate_tensor, first_input, input_tensor, *_flat_weights = ctx.saved_tensors

        hidden_size = int(grad_output.shape[1]/num_directions)
        grad_input = torch.zeros_like(first_input)
        param_num = 4 if bias else 2
        grad_flat_weights = []
        for w in _flat_weights:
            grad_flat_weights.append(torch.zeros_like(w))
        for l in reversed(range(num_layers)):
            ggate_scale_max = scale[l*4+1].add(-2).clone()
            gx_scale_max = scale[l*4+2].add(-2).clone()
            gh_scale_max = scale[l*4+3].add(-2).clone()
            if l is 0:
                input = first_input
            else:
                input = input_tensor[l-1]
            if dropout > 0 and l != num_layers-1:
                grad_output.mul_(dropout_tensor[l]/(1-dropout))
            for j in reversed(range(num_directions)):
                if j is 1:
                    grad_output_temp = torch.empty_like(grad_output)
                for k in reversed(range(seq_len)):
                    ggate_scale = scale[l*4+1].clone()
                    gx_scale = scale[l*4+2].clone()
                    gh_scale = scale[l*4+3].clone()
                    seq_idx = (seq_len-1)*j+(1-2*j)*k
                    start_idx = batch_sizes[:seq_idx].sum()
                    current_batch_size = batch_sizes[seq_idx]
                    if j is 1:
                        gate = gate_tensor[l*num_directions+j,:,batch_sizes.sum()-start_idx-current_batch_size:batch_sizes.sum()-start_idx,:]
                    else:
                        gate = gate_tensor[l*num_directions+j,:,start_idx:start_idx+current_batch_size,:]
                    rr, zz, nn, h, flat_h = gate[0], gate[1], gate[2], gate[3], gate[4]
                    alpha = grad_hidden[num_directions*l+j][:current_batch_size].add(grad_output[start_idx:start_idx+current_batch_size,hidden_size*j:hidden_size*(j+1)])
                    gn = alpha * (1 - zz) * (1 - nn * nn)
                    gz = alpha * (h - nn) * zz * (1 - zz)
                    gr = gn * flat_h * rr * (1 - rr)

                    if error_quant is None:
                        g_gate = torch.cat((gr,gz,gn),dim=1)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[start_idx:start_idx+current_batch_size]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])))
                            else:
                                grad_output[start_idx:start_idx+current_batch_size].mul_(0).add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        else:
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                        g_gate[:,2*hidden_size:].mul_(rr)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j][:current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz
                    else:
                        g_gate = error_quant.quantize(torch.cat((gr,gz,gn),dim=1), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+0].add_(g_gate.transpose(1,0).matmul(input[start_idx:start_idx+current_batch_size]))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+2].add_(g_gate.sum(dim=0))
                        if l is not 0:
                            if j is 1:
                                grad_output_temp[start_idx:start_idx+current_batch_size] = g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])
                            elif num_directions is 2:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_output_temp[start_idx:start_idx+current_batch_size].add(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0])), gx_scale)
                            else:
                                grad_output[start_idx:start_idx+current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]), gx_scale)
                        else:   
                            grad_input[start_idx:start_idx+current_batch_size].add_(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+0]))
                            if j is 0:
                                grad_input[start_idx:start_idx+current_batch_size] = error_quant.quantize(grad_input[start_idx:start_idx+current_batch_size], gx_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        ggate_scale = scale[l*4+1].clone()
                        g_gate[:,2*hidden_size:] = error_quant.quantize(g_gate[:,2*hidden_size:].mul(rr), ggate_scale)
                        grad_flat_weights[param_num*num_directions*l+param_num*j+1].add_(g_gate.transpose(1,0).matmul(h))
                        if bias:
                            grad_flat_weights[4*num_directions*l+4*j+3].add_(g_gate.sum(dim=0))
                        grad_hidden[num_directions*l+j][:current_batch_size] = error_quant.quantize(g_gate.matmul(_flat_weights[param_num*num_directions*l+param_num*j+1]) + alpha * zz, gh_scale)
                        ggate_scale_max = torch.max(ggate_scale_max, ggate_scale)
                        gx_scale_max = torch.max(gx_scale_max, gx_scale)
                        gh_scale_max = torch.max(gh_scale_max, gh_scale)
            scale[l*4+1].data = ggate_scale_max.data
            scale[l*4+2].data = gx_scale_max.data
            scale[l*4+3].data = gh_scale_max.data

        return (grad_input, None, None, None, None, None, None, None, None) + tuple(grad_flat_weights)

class GRU(torch.nn.GRU):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, batch_first = False, dropout = 0., bidirectional = False):
        super().__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        for i in range(num_layers*4):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, input, hx=None):  # noqa: F811
        if type(input) is tuple:
            input, _ = input
        scale = []
        for i in range(self.num_layers*4):
            scale.append(getattr(self, 'scale'+str(i+1)))

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = gru_func.apply(input, scale, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, self.batch_first, *tuple(self._flat_weights))
        else:
            result = packed_gru_func.apply(input, scale, batch_sizes, hx, self.bias, self.num_layers, self.dropout, self.training, self.bidirectional, *tuple(self._flat_weights))

        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


class ConvBn2d(torch.nn.Module):
    def __init__(self, conv, bn, dual=[False, False], fixed_scale=[None, None], last=False, tracking=[True, True]):
        super().__init__()
        self.cb_conv = conv
        self.cb_bn = bn
        self.cb_bn.momentum = 0.01
        self.last = last
        self.merge = False
        self.qtrain = False
        self.qblock = QBlock(dual, fixed_scale, tracking, bn=True)
        self.qfused_weight = torch.empty_like(self.cb_conv.weight)
        self.qfused_initialized = False

    def forward(self, inputs):
        if type(inputs) is not tuple:
            a, b = inputs, torch.zeros(1).to(inputs.device)
        else:
            a, b = inputs
        if self.merge:
            a = self.cb_conv(a)
        else:
            if self.qtrain:
                from .major import weight_quant, hysteresis_update
                with torch.no_grad():
                    div_val = torch.sqrt(self.cb_bn.running_var.add(self.cb_bn.eps))
                    mul_val = self.cb_bn.weight.div(div_val)
                    add_val = self.cb_bn.bias.add(-self.cb_bn.running_mean.mul(mul_val))
                    fused_weight = self.cb_conv.weight.mul(mul_val.reshape(-1,1,1,1))
                    if hysteresis_update is False or self.qfused_initialized is False:
                        self.qfused_weight.data = weight_quant.quantize(fused_weight.clone()).data
                        self.qfused_initialized = True
                    else:
                        self.qfused_weight.data = weight_quant.hysteresis(self.qfused_weight, fused_weight.clone()).data
                    err_1 = self.qfused_weight.add(-fused_weight).div(mul_val.reshape(-1,1,1,1))
                    self.cb_conv.weight.data = self.cb_conv.weight.add(err_1).data
            a = self.cb_conv(a)
            a = self.cb_bn(a)
            if self.qtrain:
                with torch.no_grad():
                    self.cb_conv.weight.data = self.cb_conv.weight.add(-err_1).data
        outputs = self.qblock((a,b))
        if self.last:
            return outputs[0]
        else:
            return outputs

    def merge_bn(self, from_qfused_buffer=False):
        if self.merge:
            return
        from .major import weight_quant
        div_val = torch.sqrt(self.cb_bn.running_var.add(self.cb_bn.eps))
        mul_val = self.cb_bn.weight.div(div_val)
        add_val = self.cb_bn.bias.add(-self.cb_bn.running_mean.mul(mul_val))
        # if weight_quant is None:
        if from_qfused_buffer:
            self.cb_conv.weight.data = self.qfused_weight.data
        else:
            self.cb_conv.weight.data = self.cb_conv.weight.mul(mul_val.reshape(-1,1,1,1)).data
        if self.cb_conv.bias is None:
            self.cb_conv.bias = torch.nn.Parameter(add_val)
        else:
            self.cb_conv.bias.data = self.cb_conv.bias.mul(mul_val).add(add_val).data
        # else:
        #     if from_qfused_buffer:
        #         self.cb_conv.weight.data = self.qfused_weight.data
        #     else:
        #         self.cb_conv.weight.data = weight_quant.quantize(self.cb_conv.weight.mul(mul_val.reshape(-1,1,1,1))).data
        #     if self.cb_conv.bias is None:
        #         self.cb_conv.bias = torch.nn.Parameter(weight_quant.quantize(add_val))
        #     else:
        #         self.cb_conv.bias.data = weight_quant.quantize(self.cb_conv.bias.mul(mul_val).add(add_val)).data
        self.merge = True

    def set_qtrain(self, qtrain=True):
        self.qtrain = qtrain
    
    def set_merge(self, merge=True):
        self.merge = merge 

'''
KQLayer : Kernel Quantization Layer
'''
class forward_kqblock_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, training, tracking):
        from .major import kactiv_quant
        if kactiv_quant is None:
            return input
        if not training:
            scale = scale.clone()
        if tracking:
            output = kactiv_quant.quantize(input, scale)
        else:
            output = kactiv_quant.quantize(input)
            scale.data = output.scale.clone().data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

class forward_KQBlock(torch.nn.Module):
    def __init__(self, tracking):
        super().__init__()
        self.tracking = tracking
        self.register_buffer('scale', torch.tensor([0], dtype=torch.int))
            
    def forward(self, input):
        return forward_kqblock_func.apply(input, self.scale, self.training, self.tracking)

class backward_kqblock_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, training, tracking):
        ctx.scale = scale
        ctx.training = training
        ctx.tracking = tracking
        return input

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        training = ctx.training
        tracking = ctx.tracking
        from .major import kerror_quant
        if kerror_quant is None:
            return grad_output, None, None, None
        if not training:
            scale = scale.clone()
        if tracking:
            grad_input = kerror_quant.quantize(grad_output, scale)
        else:
            grad_input = kerror_quant.quantize(grad_output)
            scale.data = grad_input.scale.clone().data
        return grad_input, None, None, None

class backward_KQBlock(torch.nn.Module):
    def __init__(self, tracking):
        super().__init__()
        self.tracking = tracking
        self.register_buffer('scale', torch.tensor([0], dtype=torch.int))
            
    def forward(self, input):
        return backward_kqblock_func.apply(input, self.scale, self.training, self.tracking)

class KQLayer(torch.nn.Module):
    def __init__(self, module=None, tracking=[True, True], quantize=[True, True]):
        super().__init__()
        self.module = module
        self.quantize = quantize
        self.fqblock = forward_KQBlock(tracking[0])
        self.bqblock = backward_KQBlock(tracking[1])

    def forward(self, input):
        if self.quantize[0]:
            input = self.fqblock(input)
        if self.module is not None:
            input = self.module(input)
        if self.quantize[1]:
            input = self.bqblock(input)
        return input

'''
OQLayer : Other Quantization Layer
'''
class forward_oqblock_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, training, dual, tracking):
        from .major import activ_quant
        if activ_quant is None:
            return input
        if not training:
            scale = scale.clone()
        forward_scale = scale.clone()
        if tracking:
            if dual:
                output = activ_quant.quantize(input.clone(), scale)
                output.add_(activ_quant.quantize(input.add(-output), forward_scale.add(-activ_quant.qformat.scale_diff)))
            else:
                output = activ_quant.quantize(input, scale)
        else:
            if dual:
                output = activ_quant.quantize(input.clone())
                output.add_(activ_quant.quantize(input.add(-output)))
            else:
                output = activ_quant.quantize(input)
            scale.data = output.scale.clone().data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class forward_OQBlock(torch.nn.Module):
    def __init__(self, dual, tracking):
        super().__init__()
        self.dual = dual
        self.tracking = tracking
        self.register_buffer('scale', torch.tensor([0], dtype=torch.int))
            
    def forward(self, input):
        return forward_oqblock_func.apply(input, self.scale, self.training, self.dual, self.tracking)

class backward_oqblock_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, training, dual, tracking):
        ctx.scale = scale
        ctx.training = training
        ctx.dual = dual
        ctx.tracking = tracking
        return input

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        training = ctx.training
        dual = ctx.dual
        tracking = ctx.tracking
        from .major import error_quant
        if error_quant is None:
            return grad_output, None, None, None, None
        if not training:
            scale = scale.clone()
        backward_scale = scale.clone()
        if tracking:
            if dual:
                grad_input = error_quant.quantize(grad_output.clone(), scale)
                grad_input.add_(error_quant.quantize(grad_output.add(-grad_input), backward_scale.add(-error_quant.qformat.scale_diff)))
            else:
                grad_input = error_quant.quantize(grad_output, scale)
        else:
            if dual:
                grad_input = error_quant.quantize(grad_output.clone())
                grad_input.add_(error_quant.quantize(grad_output.add(-grad_input)))
            else:
                grad_input = error_quant.quantize(grad_output)
            scale.data = grad_input.scale.clone().data
        return grad_input, None, None, None, None

class backward_OQBlock(torch.nn.Module):
    def __init__(self, dual, tracking):
        super().__init__()
        self.dual = dual
        self.tracking = tracking
        self.register_buffer('scale', torch.tensor([0], dtype=torch.int))
            
    def forward(self, input):
        return backward_oqblock_func.apply(input, self.scale, self.training, self.dual, self.tracking)

class OQLayer(torch.nn.Module):
    def __init__(self, module=None, dual=[False, False], tracking=[True, True], quantize=[True, True]):
        super().__init__()
        self.module = module
        self.quantize = quantize
        self.fqblock = forward_OQBlock(dual[0], tracking[0])
        self.bqblock = backward_OQBlock(dual[1], tracking[1])

    def forward(self, input):
        if self.quantize[0]:
            input = self.fqblock(input)
        if self.module is not None:
            input = self.module(input)
        if self.quantize[1]:
            input = self.bqblock(input)
        return input

class oqadd_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, scale, training):
        from .major import activ_quant
        if activ_quant is None:
            return a.add(b)
        if not training:
            scale[0] = scale[0].clone()
            scale[1] = scale[1].clone()
        a = activ_quant.quantize(a, scale[0])
        b = activ_quant.quantize(b, scale[1])
        output = a.add(b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None, None

class OQAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(2):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, a, b):
        scale = []
        for i in range(2):
            scale.append(getattr(self, 'scale'+str(i+1)))
        return oqadd_func.apply(a, b, scale, self.training)

class oqclone_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, scale):
        b = a.clone().detach()
        if hasattr(a, 'scale') and a.scale is not None:
            b.scale = a.scale.clone().detach()
        ctx.scale = scale
        return a, b

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        scale = ctx.scale
        from .major import error_quant
        if error_quant is None:
            return grad_output1.add(grad_output2), None
        
        grad_output1 = error_quant.quantize(grad_output1, scale[0])
        grad_output2 = error_quant.quantize(grad_output2, scale[1])
        
        grad_input = grad_output1.add(grad_output2)
        return grad_input, None

class OQClone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(2):
            self.register_buffer('scale'+str(i+1), torch.tensor([0], dtype=torch.int))

    def forward(self, input):
        scale = []
        for i in range(2):
            scale.append(getattr(self, 'scale'+str(i+1)))
        return oqclone_func.apply(input, scale)
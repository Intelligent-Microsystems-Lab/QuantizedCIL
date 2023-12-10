# code from https://openreview.net/forum?id=yTbNYYcopd
# Linear layer added following Conv layer example.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd.function import  Function
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction

corrected_version = False
quantAccBits = 8

epochnr = 0
batchnr = 0
gradient_library = {}
scale_library = {}
scale_library_hist = []
quant_range_use_perc = {}

class Conv2d_LUQ(nn.Conv2d):
    """docstring for Conv2d_BF16."""

    def __init__(self, uname, *args, **kwargs):
        super(Conv2d_LUQ, self).__init__(*args, **kwargs)
        self.fullName = ''
        self.statistics = []
        self.layerIdx = 0

        self.alpha = Parameter(torch.tensor([1], dtype=torch.float32))
        self.beta = Parameter(torch.tensor([1], dtype=torch.float32))
        self.abits = 4
        self.wbits = 4

        # correction
        if corrected_version:
            self.QnW = -(2 ** (self.wbits - 1)) + 1
            self.QpW = (2 ** (self.wbits - 1)) - 1
            self.QnA = 0
            self.QpA = 2 ** self.abits - 1
        else:
            self.QnW = -(2 ** (self.wbits - 1))
            self.QpW = 2 ** (self.wbits - 1)
            self.QnA = 0
            self.QpA = (2 ** self.abits) - 1

        self.register_buffer('init_stateW', torch.zeros(1))
        self.register_buffer('init_stateA', torch.zeros(1))

        self.register_buffer('gradScaleW', torch.zeros(1))
        self.register_buffer('gradScaleA', torch.zeros(1))

        self.quantizeFwd = True
        self.quantizeBwd = True

        self.c1 = 12.1
        self.c2 = 12.2
        self.stochastic = True
        self.repeatBwd = 1
    def forward(self, input):
        if self.quantizeFwd:
            w_q, sw = UniformQuantizeSawb.apply(self.weight,self.c1,self.c2,self.QpW,self.QnW)

            if torch.min(input) < 0:
                # correction
                if corrected_version:
                    self.QnA = -(2 ** (self.abits - 1)) + 1
                    self.QpA = (2 ** (self.abits - 1) -1) 
                else:
                    self.QnA = -2 ** (self.abits - 1)

            qinput, sa = UniformQuantizeSawb.apply(input,self.c1,self.c2,self.QpA,self.QnA)

            #all
            output = F.conv2d(qinput, w_q, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

            if False and quantAccBits < 16:
                output = AccQuant.apply(output) * sw * sa
            else:
                output = output * sw * sa

        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        output = GradStochasticClippingQ.apply(output, self.quantizeBwd,self.layerIdx,self.repeatBwd)

        return output



class Linear_LUQ(nn.Linear):
    """docstring for Conv2d_BF16."""

    def __init__(self, uname, *args, **kwargs):
        super(Linear_LUQ, self).__init__(*args, **kwargs)
        self.fullName = ''
        self.statistics = []
        self.layerIdx = 0

        self.alpha = Parameter(torch.tensor([1], dtype=torch.float32))
        self.beta = Parameter(torch.tensor([1], dtype=torch.float32))
        self.abits = 4
        self.wbits = 4

        # correction
        if corrected_version:
            self.QnW = -(2 ** (self.wbits - 1) -1)
            self.QpW = 2 ** (self.wbits - 1) - 1
            self.QnA = 0
            self.QpA = 2 ** self.abits - 1
        else:
            self.QnW = -2 ** (self.wbits - 1)
            self.QpW = 2 ** (self.wbits - 1)
            self.QnA = 0
            self.QpA = 2 ** self.abits - 1

        self.register_buffer('init_stateW', torch.zeros(1))
        self.register_buffer('init_stateA', torch.zeros(1))

        self.register_buffer('gradScaleW', torch.zeros(1))
        self.register_buffer('gradScaleA', torch.zeros(1))

        self.quantizeFwd = True
        self.quantizeBwd = True

        self.c1 = 12.1
        self.c2 = 12.2
        self.stochastic = True
        self.repeatBwd = 1
    def forward(self, input):

        if self.quantizeFwd:
            w_q, sw = UniformQuantizeSawb.apply(self.weight,self.c1,self.c2,self.QpW,self.QnW)

            if torch.min(input) < 0:
                # correction
                if corrected_version:
                    self.QnA = -(2 ** (self.abits - 1) -1)
                    self.QpA = (2 ** (self.abits - 1) -1) 
                else:
                    self.QnA = -2 ** (self.abits - 1)

            qinput, sa = UniformQuantizeSawb.apply(input,self.c1,self.c2,self.QpA,self.QnA)



            # all
            output = F.linear(qinput, w_q, self.bias,)

            if False and quantAccBits < 16:
                output = AccQuant.apply(output) * sw * sa
            else:
                output = output * sw * sa

        else:
            output = F.linear(input, self.weight, self.bias,)

        # try:
        #     assert torch.unique(w_q).shape[0] <= 16
        #     assert torch.unique(qinput).shape[0] <= 16
        # except:
        #     import pdb; pdb.set_trace()

        output = GradStochasticClippingQ.apply(output, self.quantizeBwd,self.layerIdx,self.repeatBwd)

        # if torch.isnan(output).any():
        #     import pdb; pdb.set_trace()

        return {'logits': output}


class UniformQuantizeSawb(InplaceFunction):

    @staticmethod
    def forward(ctx, input,c1,c2,Qp, Qn ):

        output = input.clone()

        with torch.no_grad():
            clip = (c1*torch.sqrt(torch.mean(input**2))) - (c2*torch.mean(input.abs()))
            scale = 2*clip / (Qp - Qn)
            output.div_(scale + 1e-5)
            output.clamp_(Qn, Qp).round_()
            # output.mul_(scale)
        return output, scale

    @staticmethod
    def backward(ctx, grad_output, g_s):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None


class AccQuant(InplaceFunction):

    @staticmethod
    def forward(ctx, input):
        output = input.clone()

        with torch.no_grad():
            n = 2**quantAccBits / 2 - 1
            output = torch.clamp(output, -n, n)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input



class GradStochasticClippingQ(Function):

    @staticmethod
    def forward(ctx, x, quantizeBwd,layerIdx,repeatBwd):
        ctx.save_for_backward(torch.tensor(quantizeBwd),torch.tensor(layerIdx),torch.tensor(repeatBwd))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        quant, layerIdx,repeatBwd = ctx.saved_tensors
        if quant:
            out = []
            for i in range(repeatBwd):

                # correction take max abs
                if corrected_version:
                    mx = torch.max(torch.abs(grad_output))
                else:
                    mx = torch.max(grad_output)
                bits = 3
                if corrected_version:
                    alpha = mx / 2**(2**bits - 2)
                else:
                    alpha = mx / (2**(2**bits-1)-1)

                alphaEps = alpha * torch.rand(grad_output.shape,device=grad_output.device)

                grad_abs = grad_output.abs()

                grad_input = torch.where(grad_abs < alpha , alpha*torch.sign(grad_output), grad_output)
                grad_input = torch.where(grad_abs < alphaEps, torch.tensor([0], dtype=torch.float32,device=grad_output.device), grad_input)

                grad_inputQ = grad_input.clone()
                noise = (2 ** torch.floor(torch.log2((grad_inputQ.abs() / alpha)) )) * grad_inputQ.new(grad_inputQ.shape).uniform_(-0.5,0.5)
                grad_inputQ = 2 ** torch.floor(torch.log2( ((grad_inputQ.abs() / alpha) + noise) *4/3 ) ) * alpha

                grad_inputQ =  torch.sign(grad_input) * torch.where(grad_inputQ < (alpha * (2 ** torch.floor(torch.log2(((grad_input.abs()/alpha)) )))),alpha *  (2 ** torch.floor(torch.log2(((grad_input.abs()/alpha)  ) ))), grad_inputQ)
                grad_inputQ = torch.where(grad_input == 0, torch.tensor([0], dtype=torch.float,device=grad_output.device), grad_inputQ)

                out.append(grad_inputQ)


            # print('h')
            # if torch.isnan(out[0]).any():
            #     import pdb; pdb.set_trace()

            grad_input = sum(out) / repeatBwd
            if False and quantAccBits < 16:
                grad_inputq = AccQuant.apply(grad_input)
            else:
                grad_inputq = grad_input 

            # global batchnr
            # global epochnr
            # if batchnr==0 and epochnr%50==0:
            #     global gradient_library
            #     global current_uname
            #     if current_uname not in gradient_library:
            #         gradient_library[current_uname] = {"gradnoq": [], "gradq": [], "gradqacc": []}
            #     gradient_library[current_uname]["gradnoq"].append(grad_output.detach().cpu().numpy())
            #     gradient_library[current_uname]["gradq"].append(grad_input.detach().cpu().numpy())
            #     gradient_library[current_uname]["gradqacc"].append(grad_inputq.detach().cpu().numpy())

        else:

            grad_inputq = grad_output

        # assert torch.unique(grad_input).shape[0] <= 16
        return grad_inputq,None, None,None



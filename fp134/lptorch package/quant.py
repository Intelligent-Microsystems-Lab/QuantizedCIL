import torch
from .major import linear_quantize, custom_fp_quantize, fp_quantize, linear_hysteresis, custom_fp_hysteresis, fp_hysteresis

class qformat():
    def __init__(self, format_type):
        self._type = format_type

    def get_type(self):
        return self._type

class linear_format(qformat):
    def __init__(self, bit_num):
        super().__init__('linear')
        self.bit_num = bit_num
        self.scale_diff = bit_num-1

class custom_fp_format(qformat):
    def __init__(self, man):
        super().__init__('custom_fp')
        self.man = torch.tensor(man, dtype=torch.int)
        self.scale_diff = man[0]

class fp_format(qformat):
    def __init__(self, exp_bit, man_bit, bias=None):
        super().__init__('fp')
        self.exp_bit = exp_bit
        self.man_bit = man_bit
        if bias is None:
            self.bias = (1 << (exp_bit-1)) - 1
        else:
            self.bias = bias

class quant():
    def __init__(self, qformat, room=0, tracking=True, stochastic=False, ch_wise=False):
        """ use accurate scale if tracking is False  """
        self._type = qformat.get_type()
        self.qformat = qformat
        self.room = room
        self.tracking = tracking
        self.stochastic = stochastic
        self.ch_wise = ch_wise

    def quantize(self, input, scale=None):
        if self.tracking is False:
            scale = None
        if self._type == 'linear':
            return linear_quantize(input, scale, self.qformat.bit_num, self.room, self.stochastic, self.ch_wise)
        elif self._type == 'custom_fp':
            return custom_fp_quantize(input, scale, self.qformat.man.to(input.device), self.room, self.stochastic, self.ch_wise)
        elif self._type == 'fp':
            return fp_quantize(input, self.qformat.exp_bit, self.qformat.man_bit, self.qformat.bias, self.stochastic)
    
    def hysteresis(self, pre_input, input, scale=None):
        if self.tracking is False:
            scale = None
        if self._type == 'linear':
            return linear_hysteresis(pre_input, input, scale, self.qformat.bit_num, self.room, self.ch_wise)
        elif self._type == 'custom_fp':
            return custom_fp_hysteresis(pre_input, input, scale, self.qformat.man.to(input.device), self.room, self.ch_wise)
        elif self._type == 'fp':
            return fp_hysteresis(pre_input, input, self.qformat.exp_bit, self.qformat.man_bit, self.qformat.bias)
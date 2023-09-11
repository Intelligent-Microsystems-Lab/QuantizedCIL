# IMSL Lab - University of Notre Dame | University of St Andrews
# Author: Clemens JS Schaefer | Martin Schiemer
# Quantized training.


from flax.linen.module import compact
import jax.numpy as jnp
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module
import jax
from jax import eval_shape
from jax import lax
from jax.core import ShapedArray
from jax.lax import conv_general_dilated_patches
import numpy as np
from flax.linen.linear import (
    canonicalize_padding,
    _conv_dimension_numbers,
    PaddingLike,
    PrecisionLike,
    default_kernel_init,
    ConvGeneralDilatedT
)


from quant_jax import flinearq
from hadamard import make_hadamard

ModuleDef = Any
Array = Any
DotGeneralT = Callable[..., Array]

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?


class Dense_Ours(Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.zeros_init()
  )
  # Deprecated. Will be removed.
  dot_general: DotGeneralT = lax.dot_general
  dot_general_cls: Any = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param(
        'kernel',
        self.kernel_init,
        (jnp.shape(inputs)[-1], self.features),
        self.param_dtype,
    )
    if self.use_bias:
      bias = self.param(
          'bias', self.bias_init, (self.features,), self.param_dtype
      )
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(
        inputs, kernel, bias, dtype=self.dtype)

    h1 = self.variable('batch_stats', 'h1', make_hadamard, self.features)
    h2 = self.variable('batch_stats', 'h2', make_hadamard, inputs.shape[0]) 

    rng = self.make_rng('stoch')

    y = flinearq(inputs, kernel, h1.value, h2.value, rng)

    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class Conv_Ours(Module):

  features: int
  kernel_size: Sequence[int]
  strides: Union[None, int, Sequence[int]] = 1
  padding: PaddingLike = 'SAME'
  input_dilation: Union[None, int, Sequence[int]] = 1
  kernel_dilation: Union[None, int, Sequence[int]] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  mask: Optional[Array] = None
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.zeros_init()
  )
  # Deprecated. Will be removed.
  conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated
  conv_general_dilated_cls: Any = None

  @property
  def shared_weights(self) -> bool:
    return True

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
     inputs: input data with dimensions (*batch_dims, spatial_dims...,
      features). This is the channels-last convention, i.e. NHWC for a 2d
      convolution and NDHWC for a 3D convolution. Note: this is different from
      the input convention used by `lax.conv_general_dilated`, which puts the
      spatial dimensions last.
      Note: If the input has more than 1 batch dimension, all batch dimensions
      are flattened into a single dimension for the convolution and restored
      before returning.  In some cases directly vmap'ing the layer may yield
      better performance than this default flattening approach.  If the input
      lacks a batch dimension it will be added for the convolution and removed
      n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """

    if isinstance(self.kernel_size, int):
      raise TypeError(
          'Expected Conv kernel_size to be a'
          ' tuple/list of integers (eg.: [3, 3]) but got'
          f' {self.kernel_size}.'
      )
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(
        x: Optional[Union[int, Sequence[int]]]
    ) -> Tuple[int, ...]:
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
          num_batch_dimensions:
      ]
      inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      kernel_size_dilated = [
          (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: List[Tuple[int, int]] = [(0, 0)]
      pads = (
          zero_pad
          + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
          + [(0, 0)]
      )
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError(
            'Causal padding is only implemented for 1D convolutions.'
        )
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    in_features = jnp.shape(inputs)[-1]

    if self.shared_weights:
      # One shared convolutional kernel for all pixels in the output.
      assert in_features % self.feature_group_count == 0
      kernel_shape = kernel_size + (
          in_features // self.feature_group_count,
          self.features,
      )

    else:
      if self.feature_group_count != 1:
        raise NotImplementedError(
            '`lax.conv_general_dilated_local` does not support '
            f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
        )

      # Need to know the spatial output shape of a standard convolution to
      # create the unshared convolution kernel.
      if self.conv_general_dilated_cls is not None:
        conv_general_dilated = self.conv_general_dilated_cls()
      else:
        conv_general_dilated = self.conv_general_dilated
      conv_output_shape = eval_shape(
          lambda lhs, rhs: conv_general_dilated(
              lhs=lhs,
              rhs=rhs,
              window_strides=strides,
              padding=padding_lax,
              dimension_numbers=dimension_numbers,
              lhs_dilation=input_dilation,
              rhs_dilation=kernel_dilation,
          ),
          inputs,
          ShapedArray(kernel_size + (in_features,
                      self.features), inputs.dtype),
      ).shape

      # One (unshared) convolutional kernel per each pixel in the output.
      kernel_shape = conv_output_shape[1:-1] + (
          np.prod(kernel_size) * in_features,
          self.features,
      )

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError(
          'Mask needs to have the same shape as weights. '
          f'Shapes are: {self.mask.shape}, {kernel_shape}'
      )

    kernel = self.param(
        'kernel', self.kernel_init, kernel_shape, self.param_dtype
    )

    if self.mask is not None:
      kernel *= self.mask

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)
      else:
        # One bias weight per output entry, unshared betwen pixels.
        bias_shape = conv_output_shape[1:]

      bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
    else:
      bias = None

    h1 = self.variable('batch_stats', 'h1', make_hadamard, self.features)
    h2 = self.variable('batch_stats', 'h2', make_hadamard,
                       inputs.shape[1] * inputs.shape[2])

    inputs, kernel, bias = promote_dtype(
        inputs, kernel, bias, dtype=self.dtype)
    if self.shared_weights:
      if self.conv_general_dilated_cls is not None:
        conv_general_dilated = self.conv_general_dilated_cls()
      else:
        conv_general_dilated = self.conv_general_dilated

      # [kH, kW, inC, outC] -> [outC, inC, kH, kW]
      w_q = jnp.transpose(kernel, (3, 2, 0, 1))
      w_q = jnp.reshape(w_q, (w_q.shape[0], -1)).transpose()

      qinputs = conv_general_dilated_patches(inputs,
                                             kernel_size,
                                             strides,
                                             padding_lax,
                                             lhs_dilation=input_dilation,
                                             rhs_dilation=kernel_dilation,
                                             dimension_numbers=dimension_numbers,
                                             precision=self.precision,)
      qinputs = jnp.reshape(qinputs, (qinputs.shape[0], -1, qinputs.shape[-1]))

      h1_exp = jnp.repeat(jnp.expand_dims(h1.value, 0),
                          qinputs.shape[0], axis=0)
      h2_exp = jnp.repeat(jnp.expand_dims(h2.value, 0),
                          qinputs.shape[0], axis=0)

      rng = self.make_rng('stoch')
      rng = jax.random.split(rng, qinputs.shape[0])

      y = jax.vmap(flinearq)(qinputs, jnp.repeat(jnp.expand_dims(
          w_q, 0), qinputs.shape[0], axis=0), h1_exp, h2_exp, rng)

      y = jnp.reshape(y, (y.shape[0], int(
          inputs.shape[1] / strides[0]), int(inputs.shape[2] / strides[1]),
          self.features))

    else:
      raise Exception('not implemented for weights not shared...')
      y = lax.conv_general_dilated_local(
          lhs=inputs,
          rhs=kernel,
          window_strides=strides,
          padding=padding_lax,
          filter_shape=kernel_size,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          precision=self.precision,
      )

    if self.use_bias:
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y

# IMSL Lab - University of Notre Dame | University of St Andrews
# Author: Clemens JS Schaefer | Martin Schiemer
# Quantized training.

import jax
import jax.numpy as jnp
from jax import custom_vjp


from hadamard import biggest_power2_factor


quantBits = 4
quantAccBits = 8
quantWgtStoreBits = 8
quantBlockSize = 32


@custom_vjp
def flinearq(x, w, h1, h2, rng):
  # print('-----')
  # print(h2.shape)
  # print(x.shape[0])
  # print(w.shape[1])
  return jnp.dot(x, w)


def flinearq_fwd(x, w, h1, h2, rng):
  return flinearq(x, w, h1, h2, rng), (x, w, h1, h2, rng)


def flinearq_bwd(res, g):
  x, w, h1, h2, rng = res

  g1h = jnp.dot(g, h1)

  wh = jnp.dot(h1, jnp.transpose(w))

  grad_x = jnp.dot(g1h, wh)

  xh = jnp.dot(jnp.transpose(x), h2)

  g2h = jnp.dot(h2, g)

  grad_w = jnp.dot(xh, g2h)

  return grad_x * 1 / biggest_power2_factor(h1.shape[0]), \
      grad_w * 1 / biggest_power2_factor(h2.shape[0]), None, None, None


flinearq.defvjp(flinearq_fwd, flinearq_bwd)

# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer and Martin Schiemer
# Quantized training.
import math
import numpy as np
import scipy


def hadamard(n, dtype=int):
  # from https://github.com/scipy/scipy/blob/v1.11.1/scipy/linalg/_special_matrices.py#L319-L373

  if n < 1:
    lg2 = 0
  else:
    lg2 = int(math.log(n, 2))
  if 2 ** lg2 != n:
    raise ValueError("n must be an positive integer, and n must be "
                     "a power of 2")

  H = np.array([[1]], dtype=dtype)

  # Sylvester's construction
  for i in range(0, lg2):
    H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

  return H


def biggest_power2_factor(n):
  factors = []
  for i in range(1, n + 1):
    if n % i == 0:
      if math.log(i, 2).is_integer():
        factors.append(i)
  return max(factors)


def make_hadamard(size_m):
  biggest_pow2 = biggest_power2_factor(size_m)
  return scipy.linalg.block_diag(*[hadamard(biggest_pow2)] * int(size_m / biggest_pow2))

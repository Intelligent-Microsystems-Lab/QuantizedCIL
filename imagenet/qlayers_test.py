# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for the ImageNet example using fake data for quick perf results.

This script doesn't need the dataset, but it needs the dataset metadata.
That can be fetched with the script `flax/tests/download_dataset_metadata.sh`.
"""

import time

from absl.testing import absltest
from flax.testing import Benchmark
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


from qlayers import Conv_Ours, Dense_Ours

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class UnitTestCustomQuantLayer(Benchmark):
  """Runs ImageNet using fake data for quickly measuring performance."""

  def test_eq_conv_fwd(self):
    key = jax.random.PRNGKey(8627169)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    variables = Conv_Ours(16, (3, 3)).init(
        {'params': subkey1, 'stoch': subkey2}, jnp.zeros((128, 32, 32, 3)))

    key, subkey1, subkey2 = jax.random.split(key, 3)
    data = jax.random.normal(subkey1, (128, 32, 32, 3),)

    std_flax = nn.Conv(16, (3, 3)).apply(
        {'params': variables['params']}, inputs=data, rngs={'stoch': subkey2})

    start_time = time.time()
    ours = Conv_Ours(16, (3, 3)).apply(
        variables, inputs=data, rngs={'stoch': subkey2})
    benchmark_time = time.time() - start_time

    np.testing.assert_allclose(std_flax, ours)

    self.report_wall_time(benchmark_time)
    self.report_extras({
        'description': 'fwd conv custom quant conv against flax standard.',
        'model_name': 'conv',
        'parameters': 'features=64,kernel_size=(3,3)',
    })

  def test_eq_conv_bwd(self):
    key = jax.random.PRNGKey(8627169)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    variables = Conv_Ours(16, (3, 3)).init(
        {'params': subkey1, 'stoch': subkey2}, jnp.zeros((128, 32, 32, 3)))

    key, subkey1, subkey2 = jax.random.split(key, 3)
    data = jax.random.normal(subkey1, (128, 32, 32, 3),)
    targets = jax.random.normal(subkey2, (128, 32, 32, 16),)

    def loss_fn1(w):
      x = nn.Conv(16, (3, 3)).apply(
          {'params': w}, inputs=data, rngs={'stoch': subkey2})
      return jnp.mean((x - targets)**2)

    def loss_fn2(w):
      x = Conv_Ours(16, (3, 3)).apply(
          {'params': w, 'batch_stats': variables['batch_stats']}, inputs=data,
          rngs={'stoch': subkey2})
      return jnp.mean((x - targets)**2)

    g1 = jax.grad(loss_fn1)(variables['params'])

    start_time = time.time()
    g2 = jax.grad(loss_fn2)(variables['params'])
    benchmark_time = time.time() - start_time

    np.testing.assert_allclose(
        g1['kernel'], g2['kernel'], rtol=1e-5, atol=1e-7)

    self.report_wall_time(benchmark_time)
    self.report_extras({
        'description': 'bwd conv custom quant conv against flax standard.',
        'model_name': 'conv',
        'parameters': 'features=64,kernel_size=(3,3)',
    })

  def test_eq_jit_conv_bwd(self):
    key = jax.random.PRNGKey(8627169)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    variables = Conv_Ours(16, (3, 3)).init(
        {'params': subkey1, 'stoch': subkey2}, jnp.zeros((128, 32, 32, 3)))

    key, subkey1, subkey2 = jax.random.split(key, 3)
    data = jax.random.normal(subkey1, (128, 32, 32, 3),)
    targets = jax.random.normal(subkey2, (128, 32, 32, 16),)

    @jax.jit
    def loss_fn1(w):
      x = nn.Conv(16, (3, 3)).apply(
          {'params': w}, inputs=data, rngs={'stoch': subkey2})
      return jnp.mean((x - targets)**2)

    @jax.jit
    def loss_fn2(w):
      x = Conv_Ours(16, (3, 3)).apply(
          {'params': w, 'batch_stats': variables['batch_stats']}, inputs=data,
          rngs={'stoch': subkey2})
      return jnp.mean((x - targets)**2)

    g1 = jax.grad(loss_fn1)(variables['params'])

    start_time = time.time()
    g2 = jax.grad(loss_fn2)(variables['params'])
    benchmark_time = time.time() - start_time

    np.testing.assert_allclose(
        g1['kernel'], g2['kernel'], rtol=1e-5, atol=1e-7)

    self.report_wall_time(benchmark_time)
    self.report_extras({
        'description': 'jit bwd conv custom quant conv against flax standard.',
        'model_name': 'conv',
        'parameters': 'features=64,kernel_size=(3,3)',
    })

  def test_eq_dense_fwd(self):
    key = jax.random.PRNGKey(8627169)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    variables = Dense_Ours(64,).init(
        {'params': subkey1, 'stoch': subkey2}, jnp.zeros((128, 32)))

    key, subkey1, subkey2 = jax.random.split(key, 3)
    data = jax.random.normal(subkey1, (128, 32),)

    std_flax = nn.Dense(64,).apply(
        {'params': variables['params']}, inputs=data, rngs={'stoch': subkey2})

    start_time = time.time()
    ours = Dense_Ours(64).apply(
        variables, inputs=data, rngs={'stoch': subkey2})
    benchmark_time = time.time() - start_time

    np.testing.assert_allclose(std_flax, ours)

    self.report_wall_time(benchmark_time)
    self.report_extras({
        'description': 'fwd dense custom quant dense against flax standard.',
        'model_name': 'dense',
        'parameters': 'features=64',
    })

  def test_eq_dense_bwd(self):
    key = jax.random.PRNGKey(8627169)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    variables = Dense_Ours(64,).init(
        {'params': subkey1, 'stoch': subkey2}, jnp.zeros((128, 32)))

    key, subkey1, subkey2 = jax.random.split(key, 3)
    data = jax.random.normal(subkey1, (128, 32),)
    targets = jax.random.normal(subkey2, (128, 64),)

    def loss_fn1(w):
      x = nn.Dense(64).apply(
          {'params': w}, inputs=data, rngs={'stoch': subkey2})
      return jnp.mean((x - targets)**2)

    def loss_fn2(w):
      x = Dense_Ours(64).apply(
          {'params': w, 'batch_stats': variables['batch_stats']}, inputs=data,
          rngs={'stoch': subkey2})
      return jnp.mean((x - targets)**2)

    g1 = jax.grad(loss_fn1)(variables['params'])

    start_time = time.time()
    g2 = jax.grad(loss_fn2)(variables['params'])
    benchmark_time = time.time() - start_time

    np.testing.assert_allclose(
        g1['kernel'], g2['kernel'], rtol=1e-5, atol=1e-7)

    self.report_wall_time(benchmark_time)
    self.report_extras({
        'description': 'bwd dense custom quant dense against flax standard.',
        'model_name': 'dense',
        'parameters': 'features=64',
    })

  def test_eq_jit_dense_bwd(self):
    key = jax.random.PRNGKey(8627169)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    variables = Dense_Ours(64,).init(
        {'params': subkey1, 'stoch': subkey2}, jnp.zeros((128, 32)))

    key, subkey1, subkey2 = jax.random.split(key, 3)
    data = jax.random.normal(subkey1, (128, 32),)
    targets = jax.random.normal(subkey2, (128, 64),)

    @jax.jit
    def loss_fn1(w):
      x = nn.Dense(64).apply(
          {'params': w}, inputs=data, rngs={'stoch': subkey2})
      return jnp.mean((x - targets)**2)

    @jax.jit
    def loss_fn2(w):
      x = Dense_Ours(64).apply(
          {'params': w, 'batch_stats': variables['batch_stats']}, inputs=data,
          rngs={'stoch': subkey2})
      return jnp.mean((x - targets)**2)

    g1 = jax.grad(loss_fn1)(variables['params'])

    start_time = time.time()
    g2 = jax.grad(loss_fn2)(variables['params'])
    benchmark_time = time.time() - start_time

    np.testing.assert_allclose(
        g1['kernel'], g2['kernel'], rtol=1e-5, atol=1e-7)

    self.report_wall_time(benchmark_time)
    self.report_extras({
        'description': 'jit bwd dense custom quant dense against flax.',
        'model_name': 'dense',
        'parameters': 'features=64',
    })


if __name__ == '__main__':
  absltest.main()

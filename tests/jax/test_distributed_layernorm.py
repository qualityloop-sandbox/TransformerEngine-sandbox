# This file was modified for portability to AMDGPU
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import warnings
import pytest
from packaging import version

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental.custom_partitioning import BATCHING, CompoundFactor, SdyShardingRule
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from distributed_test_base import generate_configs, generate_collectives_count
from distributed_test_base import compare_ops
from utils import pytest_parametrize_wrapper

from transformer_engine.jax import autocast
from transformer_engine.common import recipe
from transformer_engine.jax.cpp_extensions.normalization import NormFwdPrimitive, NVTE_Norm_Type
from transformer_engine.jax.layernorm import layernorm
from transformer_engine.jax.quantize import (
    QuantizeLayout,
    QuantizerFactory,
    ScalingMode,
    is_fp8_available,
)
from transformer_engine.jax.util import get_jnp_float8_e4m3_type, get_jnp_float8_e5m2_type


DTYPES = [jnp.bfloat16, jnp.float32]

NORM_INPUT_SHAPES = {
    "L0": [[64, 64]],
    "L1": [[64, 64]],
    "L2": [[64, 64]],
}

is_fp8_supported, reason = is_fp8_available()
is_mxfp8_supported, reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

SUPPORTED_RECIPES = []
if is_fp8_supported:
    SUPPORTED_RECIPES.append(pytest.param(recipe.DelayedScaling(), id="DelayedScaling"))
    SUPPORTED_RECIPES.append(pytest.param(recipe.Float8CurrentScaling(), id="CurrentScaling"))
if is_mxfp8_supported:
    SUPPORTED_RECIPES.append(pytest.param(recipe.MXFP8BlockScaling(), id="MXFP8BlockScaling"))


class TestDistributedLayernorm:

    def generate_inputs(self, shape, mesh_resource, dtype, shard_weights):
        weight_shape = (shape[-1],)

        x = random.normal(random.PRNGKey(1124), shape, dtype=dtype)
        gamma = jnp.ones(weight_shape, dtype=dtype)
        beta = jnp.ones(weight_shape, dtype=dtype)

        if len(shape) == 2:
            x_pspec = PartitionSpec(mesh_resource.dp_resource, None)
        elif len(shape) == 3:
            x_pspec = PartitionSpec(mesh_resource.dp_resource, None, None)
        else:
            raise NotImplementedError

        g_pspec = b_pspec = (
            PartitionSpec(mesh_resource.dp_resource) if shard_weights else PartitionSpec(None)
        )

        return (x, gamma, beta), (x_pspec, g_pspec, b_pspec)

    def generate_collectives_count_ref(
        self, mesh_resource, ln_type, shape, dtype, mesh_axes, fp8_recipe
    ):
        jax_dtype = jax.dtypes.canonicalize_dtype(dtype)
        # TODO(Phuong) is_dp_enabled = dp mesh axis size > 1
        is_dp_enabled = mesh_resource.dp_resource is not None
        is_tpsp_enabled = mesh_resource.tpsp_resource is not None
        assert ln_type in ["layernorm", "rmsnorm"]
        # loss, 1 FP32
        allreduce_total_bytes = 4 if is_dp_enabled else 0
        # dgamma and dbeta
        weight_count = 2 if ln_type == "layernorm" else 1
        allreduce_total_bytes += weight_count * shape[-1] * jax_dtype.itemsize
        return generate_collectives_count(
            allreduce=allreduce_total_bytes * int(is_dp_enabled or is_tpsp_enabled),
            allgather=0,
            other=0,
        )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper("data_shape", NORM_INPUT_SHAPES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("zero_centered_gamma", [False, True])
    @pytest_parametrize_wrapper("shard_weights", [False, True])
    @pytest_parametrize_wrapper("fp8_recipe", SUPPORTED_RECIPES)
    def test_layernorm(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        dtype,
        zero_centered_gamma,
        shard_weights,
        fp8_recipe,
    ):
        epsilon = 1e-6
        ln_type = "layernorm"
        q_dtype = get_jnp_float8_e4m3_type()

        def target_func(x, gamma, beta):
            quantizer = QuantizerFactory.create_set().x
            return jnp.mean(
                layernorm(
                    x, gamma, beta, ln_type, zero_centered_gamma, epsilon, quantizer=quantizer
                )
            )

        def ref_func(x, gamma, beta):
            x_ = jnp.asarray(x, jnp.float32)
            mean = jnp.mean(x_, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x_ - mean), axis=-1, keepdims=True)
            normed_input = (x_ - mean) * jax.lax.rsqrt(var + epsilon)
            if zero_centered_gamma:
                output = jnp.asarray(normed_input * (gamma + 1) + beta).astype(x.dtype)
            else:
                output = jnp.asarray(normed_input * gamma + beta).astype(x.dtype)
            return jnp.mean(output)

        (x, gamma, beta), (x_pspec, g_pspec, b_pspec) = self.generate_inputs(
            data_shape, mesh_resource, dtype, shard_weights
        )
        collective_count_ref = self.generate_collectives_count_ref(
            mesh_resource, ln_type, data_shape, dtype, mesh_axes, fp8_recipe
        )
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with jax.set_mesh(mesh), autocast(enabled=True, recipe=fp8_recipe, mesh_resource=mesh_resource):
            x_named_sharding = NamedSharding(mesh, x_pspec)
            g_named_sharding = NamedSharding(mesh, g_pspec)
            b_named_sharding = NamedSharding(mesh, b_pspec)
            x_ = jax.device_put(x, x_named_sharding)
            gamma_ = jax.device_put(gamma, g_named_sharding)
            beta_ = jax.device_put(beta, b_named_sharding)

            with warnings.catch_warnings(record=True) as warns:
                try:
                    compare_ops(
                        target_func,
                        ref_func,
                        [x_, gamma_, beta_],
                        collective_count_ref,
                        grad_args=(0, 1, 2),
                        metric_fwd_dtype=q_dtype,
                        metric_bwd_dtype=q_dtype,
                        in_shardings=(x_named_sharding, g_named_sharding, b_named_sharding),
                        out_shardings=(
                            None,
                            (x_named_sharding, g_named_sharding, b_named_sharding),
                        ),
                    )
                except AssertionError as err:
                    # Layernorm should still produce the correct numerical result with
                    # gamma/beta sharded. However, the collective count may not be the same
                    # when XLA is forced to unshard gamma and/or beta. We can catch
                    # and ignore that specific error here.
                    if (
                        g_pspec[-1] is None and b_pspec[-1] is None
                    ) or "Expected collective count" not in str(err):
                        raise err
                finally:
                    for w in warns:
                        assert "Enforcing no sharding of parameters hidden dim!" in str(w), (
                            "Layernorm primitive did not raise the correct warning for "
                            "unsupported sharding of gamma and/or beta"
                        )

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest_parametrize_wrapper("data_shape", NORM_INPUT_SHAPES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("shard_weights", [False, True])
    @pytest_parametrize_wrapper("fp8_recipe", SUPPORTED_RECIPES)
    def test_rmsnorm(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        dtype,
        shard_weights,
        fp8_recipe,
    ):
        epsilon = 1e-6
        ln_type = "rmsnorm"
        q_dtype = get_jnp_float8_e4m3_type()

        def target_func(x, gamma):
            quantizer = QuantizerFactory.create_set().x
            return jnp.mean(layernorm(x, gamma, None, ln_type, False, epsilon, quantizer=quantizer))

        def ref_func(x, gamma):
            x = jnp.asarray(x, jnp.float32)
            mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), dtype)
            output = y * gamma
            return jnp.mean(output)

        (x, gamma, _), (x_pspec, g_pspec, _) = self.generate_inputs(
            data_shape, mesh_resource, dtype, shard_weights
        )
        collective_count_ref = self.generate_collectives_count_ref(
            mesh_resource, ln_type, data_shape, dtype, mesh_axes, fp8_recipe
        )
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with jax.set_mesh(mesh), autocast(enabled=True, recipe=fp8_recipe, mesh_resource=mesh_resource):
            x_named_sharding = NamedSharding(mesh, x_pspec)
            g_named_sharding = NamedSharding(mesh, g_pspec)
            x_ = jax.device_put(x, x_named_sharding)
            gamma_ = jax.device_put(gamma, g_named_sharding)

            with warnings.catch_warnings(record=True) as warns:
                try:
                    compare_ops(
                        target_func,
                        ref_func,
                        [x_, gamma_],
                        collective_count_ref,
                        grad_args=(0, 1),
                        metric_fwd_dtype=q_dtype,
                        metric_bwd_dtype=q_dtype,
                        in_shardings=(x_named_sharding, g_named_sharding),
                        out_shardings=(None, (x_named_sharding, g_named_sharding)),
                    )
                except AssertionError as err:
                    # RmsNorm should still produce the correct numerical result with
                    # gamma/beta sharded. However, the collective count may not be the same
                    # when XLA is forced to unshard gamma. We can catch
                    # and ignore that specific error here.
                    if g_pspec[-1] is None or "Expected collective count" not in str(err):
                        raise err
                finally:
                    for w in warns:
                        assert "Enforcing no sharding of parameters hidden dim!" in str(w), (
                            "RmsNorm primitive did not raise the correct warning for "
                            "unsupported sharding of gamma and/or beta"
                        )


class TestNormFwdPrimitiveShardyShardingRule:
    """Unit tests for NormFwdPrimitive.shardy_sharding_rule.

    These tests call the static method directly with plain Python/JAX values
    instead of running it through jit + a real mesh, since the rule itself is
    pure metadata computation that does not touch any device.
    """

    def _call_rule(self, x_shape, norm_type, scaling_mode, quantize_layout):
        x_aval = jax.core.ShapedArray(shape=x_shape, dtype=jnp.bfloat16)
        return NormFwdPrimitive.shardy_sharding_rule(
            norm_type,
            False,  # zero_centered_gamma
            1e-6,  # epsilon
            jnp.bfloat16,  # out_dtype
            scaling_mode.value,
            quantize_layout,
            jnp.float32,  # scale_dtype
            None,  # amax_scope
            False,  # transpose_batch_sequence
            False,  # output_amax_when_no_scaling
            False,  # is_outer
            None,  # mesh
            [x_aval],  # value_types
            None,  # result_types
        )

    @pytest_parametrize_wrapper(
        "norm_type", [NVTE_Norm_Type.LayerNorm, NVTE_Norm_Type.RMSNorm]
    )
    @pytest_parametrize_wrapper("x_shape", [(64, 64), (8, 32, 64)])
    def test_no_scaling(self, norm_type, x_shape):
        rule = self._call_rule(
            x_shape, norm_type, ScalingMode.NO_SCALING, QuantizeLayout.ROWWISE
        )

        assert isinstance(rule, SdyShardingRule)
        assert len(rule.operand_mappings) == 5  # x, scale, amax, gamma, beta
        assert len(rule.result_mappings) == 7  # out, colwise_out, scale_inv x2, amax, mu, rsigma

        x_spec, scale_spec, amax_spec, gamma_spec, beta_spec = rule.operand_mappings
        (
            out_spec,
            colwise_out_spec,
            rowwise_scale_inv_spec,
            colwise_scale_inv_spec,
            result_amax_spec,
            mu_spec,
            rsigma_spec,
        ) = rule.result_mappings

        # No scaling means the output is unquantized and keeps the same spec as the input.
        assert len(x_spec) == len(x_shape)
        assert tuple(out_spec) == tuple(x_spec)

        # Scale/amax/gamma/beta are all fully-batched (single factor) dimensions.
        for spec in (scale_spec, amax_spec, gamma_spec, beta_spec, result_amax_spec):
            assert len(spec) == 1
            assert spec[0][0] == BATCHING

        # scale (operand) and rowwise scale_inv (result) refer to the same unsharded factor.
        assert tuple(scale_spec) == tuple(rowwise_scale_inv_spec)
        assert len(colwise_out_spec) == 1 and colwise_out_spec[0][0] == BATCHING
        assert len(colwise_scale_inv_spec) == 1 and colwise_scale_inv_spec[0][0] == BATCHING

        # rsigma matches all but the last (hidden) dim of x.
        assert tuple(rsigma_spec) == tuple(x_spec[:-1])
        if norm_type == NVTE_Norm_Type.LayerNorm:
            # LayerNorm's mu has the same shape/sharding as rsigma.
            assert tuple(mu_spec) == tuple(rsigma_spec)
        else:
            # RMSNorm's mu is a placeholder scalar and is not derived from x.
            assert len(mu_spec) == 1
            assert mu_spec[0][0] == BATCHING
            assert mu_spec[0] != rsigma_spec[0]

        # No scaling mode never needs extra factor sizes (no block/compound factors).
        assert rule.factor_sizes == {}

    @pytest_parametrize_wrapper(
        "scaling_mode", [ScalingMode.DELAYED_TENSOR_SCALING, ScalingMode.CURRENT_TENSOR_SCALING]
    )
    def test_tensor_scaling_modes(self, scaling_mode):
        x_shape = (64, 64)
        rule = self._call_rule(
            x_shape, NVTE_Norm_Type.LayerNorm, scaling_mode, QuantizeLayout.ROWWISE
        )

        x_spec = rule.operand_mappings[0]
        out_spec = rule.result_mappings[0]
        # Tensor-wide scaling factors keep the quantized output aligned with the input sharding.
        assert tuple(out_spec) == tuple(x_spec)
        assert rule.factor_sizes == {}

    def test_mxfp8_block_scaling_uses_compound_factor(self):
        x_shape = (64, 64)
        rule = self._call_rule(
            x_shape,
            NVTE_Norm_Type.LayerNorm,
            ScalingMode.MXFP8_1D_SCALING,
            QuantizeLayout.ROWWISE,
        )

        x_spec = rule.operand_mappings[0]
        # The last (block-scaled) dim of x is expressed as a CompoundFactor, not a plain factor,
        # since it must be split into a "logical size" factor and a "block size" factor.
        assert isinstance(x_spec[-1], CompoundFactor)
        assert "blocksize_x" in rule.factor_sizes
        assert rule.factor_sizes["blocksize_x"] == 32

    def test_requires_jax_0_5(self, monkeypatch):
        monkeypatch.setattr(jax, "__version__", "0.4.30")
        with pytest.raises(ImportError):
            self._call_rule(
                (64, 64), NVTE_Norm_Type.LayerNorm, ScalingMode.NO_SCALING, QuantizeLayout.ROWWISE
            )

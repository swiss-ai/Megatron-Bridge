"""Shared live offloading-expert views, independent of HF model family."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe.experts import OffloadingExpertsMLP
from transformers import PretrainedConfig

from megatron.bridge.models.apertus2.apertus2_mapping import build_apertus2_mapping_registry
from megatron.bridge.models.conversion import model_bridge
from megatron.bridge.models.conversion.utils import (
    _iter_model_parameters_and_buffers,
    _OffloadingExpertLinearView,
    get_module_and_param_from_name,
)
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge


pytestmark = pytest.mark.unit


class _Group:
    def __init__(self, size=1, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@pytest.fixture
def model():
    # Exercise the real kernel type without allocating its CUDA runtime buffers.
    kernel = OffloadingExpertsMLP.__new__(OffloadingExpertsMLP)
    torch.nn.Module.__init__(kernel)
    kernel.config = SimpleNamespace(
        moe_use_offloading_experts=True,
        moe_use_inplace_fp8_param=True,
        moe_use_extra_fp8_param_storage=True,
        gated_linear_unit=True,
        num_moe_experts=4,
    )
    kernel.num_local_experts = 2
    kernel.weight1 = torch.nn.Parameter(torch.arange(24, dtype=torch.bfloat16).reshape(2, 4, 3))
    kernel.weight2 = torch.nn.Parameter(torch.arange(12, dtype=torch.bfloat16).reshape(2, 3, 2))
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = kernel
    result = torch.nn.Module()
    result.decoder = torch.nn.Module()
    result.decoder.layers = torch.nn.ModuleList([layer])
    result.config = kernel.config
    return result


def test_aliases_are_live_and_writable(model):
    names = dict(_iter_model_parameters_and_buffers(model))
    prefix = "decoder.layers.0.mlp.experts"
    assert f"{prefix}.weight1" not in names
    assert f"{prefix}.weight2" not in names
    assert len(names) == 4
    module, weight = get_module_and_param_from_name(model, f"{prefix}.linear_fc1.weight1")
    assert isinstance(module, _OffloadingExpertLinearView)
    assert module.num_gemms == 2
    assert module.partition_dim == 0
    assert weight.shape == (4, 3)
    assert weight.dtype == torch.bfloat16
    with torch.no_grad():
        weight.fill_(7)
    torch.testing.assert_close(model.decoder.layers[0].mlp.experts.weight1[1], weight)
    with torch.no_grad():
        model.decoder.layers[0].mlp.experts.weight1[1].fill_(9)
    assert torch.all(weight == 9)


def test_regular_conversion_tasks_use_live_views(model, monkeypatch):
    bridge = Qwen3MoEBridge()
    group = _Group()
    groups = SimpleNamespace(tp=group, expt_tp=group, ep=group, pp=group)
    names = list(dict(_iter_model_parameters_and_buffers(model)))
    monkeypatch.setattr(bridge, "_megatron_global_param_names_all_pp_ranks", lambda _: names)
    monkeypatch.setattr(model_bridge, "_get_pg_collection_from_model", lambda _: groups)
    monkeypatch.setattr(model_bridge, "_get_pp_rank", lambda _: 0)
    monkeypatch.setattr(model_bridge, "_get_pp_group", lambda _: group)
    monkeypatch.setattr(model_bridge, "_get_ep_group", lambda _: group)
    tasks = bridge.build_conversion_tasks(PretrainedConfig(), [model])
    assert len(tasks) == 4
    for task in tasks:
        assert isinstance(task.megatron_module, _OffloadingExpertLinearView)
        assert task.param_weight.ndim == 2
        assert task.param_weight.dtype == torch.bfloat16
        assert task.mapping.is_expert


def test_ordinary_parameters_are_unchanged():
    ordinary = torch.nn.Linear(3, 4)
    actual = dict(_iter_model_parameters_and_buffers(ordinary))
    assert actual["weight"] is ordinary.weight
    assert actual["bias"] is ordinary.bias


def test_missing_master_storage_is_rejected(model):
    model.config.moe_use_extra_fp8_param_storage = False
    with pytest.raises(ValueError, match="BF16 masters"):
        list(_iter_model_parameters_and_buffers(model))


def test_missing_kernel_fails_when_offloading_is_enabled(model, monkeypatch):
    from megatron.core.transformer.moe import experts

    monkeypatch.delattr(experts, "OffloadingExpertsMLP")
    with pytest.raises(RuntimeError, match="active Megatron-Core"):
        list(_iter_model_parameters_and_buffers(model))
    # Ordinary models do not require the optional offloading kernel.
    assert len(list(_iter_model_parameters_and_buffers(torch.nn.Linear(3, 4)))) == 2


def test_global_expert_offset(model, monkeypatch):
    monkeypatch.setattr(model_bridge, "_get_pp_group", lambda _: _Group())
    monkeypatch.setattr(model_bridge, "_get_ep_group", lambda _: _Group(2, 1))
    monkeypatch.setattr(model_bridge, "get_pg_size", lambda group: group.size())
    name = "decoder.layers.0.mlp.experts.linear_fc1.weight1"
    assert model_bridge._megatron_local_name_to_global(model, model.config, name).endswith("weight3")


@pytest.mark.parametrize("family", ["apertus2", "qwen3_moe"])
def test_existing_family_mappings_export_grouped_views(model, family):
    if family == "apertus2":
        config = SimpleNamespace(
            attention_output_gate=False,
            layer_types=("full_attention",),
            moe_layer_freq=[1],
            moe_router_enable_expert_bias=False,
            num_hidden_layers=1,
            qk_layernorm=False,
            sandwich_norm=False,
            use_quantile_balancing=True,
        )
        registry = build_apertus2_mapping_registry(config)
    else:
        registry = Qwen3MoEBridge().mapping_registry()
    group = _Group()
    registry.set_process_groups_from_pg_collection(SimpleNamespace(tp=group, expt_tp=group, ep=group, pp=group))
    prefix = "decoder.layers.0.mlp.experts"
    for index in range(2):
        fc1_name = f"{prefix}.linear_fc1.weight{index}"
        module, weight = get_module_and_param_from_name(model, fc1_name)
        mapping = registry.megatron_to_hf_lookup(fc1_name)
        exported = mapping.megatron_to_hf(weight, module)
        hf_prefix = f"model.layers.0.mlp.experts.{index}"
        torch.testing.assert_close(exported[f"{hf_prefix}.gate_proj.weight"], weight[:2])
        torch.testing.assert_close(exported[f"{hf_prefix}.up_proj.weight"], weight[2:])
        fc2_name = f"{prefix}.linear_fc2.weight{index}"
        module, weight = get_module_and_param_from_name(model, fc2_name)
        mapping = registry.megatron_to_hf_lookup(fc2_name)
        exported = mapping.megatron_to_hf(weight, module)
        torch.testing.assert_close(exported[f"{hf_prefix}.down_proj.weight"], weight)

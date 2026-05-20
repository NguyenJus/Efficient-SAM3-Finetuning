"""CPU smoke tests for esam3.peft_adapters.qlora.

The real work (4-bit module swap + LoRA on quantized base) is GPU-only and
lives in tests/integration/test_peft_qlora_real.py. These tests cover:
  - registry wiring
  - schema parse for PEFTConfig(method="qlora") and QLoRAConfig
  - lazy import of bitsandbytes (module import must succeed without bnb)
  - ImportError surface when apply_qlora is called without bnb
  - _infer_quant_type_from_wrapper fallback chain (primary + legacy + error)

TDD-red status: this file is committed BEFORE the implementation lands in
Task 3. Three tests are expected to fail against the current
NotImplementedError stub and will go green when Task 3 lands:
  * test_apply_qlora_raises_helpful_importerror_when_bnb_missing
  * test_save_qlora_raises_when_no_peft_model
  * test_load_qlora_raises_when_peft_model_already_set
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from esam3._registry import lookup
from esam3.config.schema import PEFTConfig, QLoRAConfig
from tests.fixtures.tiny_sam3_lora_stub import make_stub_wrapper


def test_registry_lookup() -> None:
    """apply_qlora is registered under ('peft', 'qlora')."""
    from esam3.peft_adapters.qlora import apply_qlora

    assert lookup("peft", "qlora") is apply_qlora


def test_schema_qlora_method_defaults() -> None:
    """PEFTConfig(method='qlora') validates with default QLoRAConfig."""
    cfg = PEFTConfig(method="qlora")
    assert cfg.method == "qlora"
    assert cfg.qlora.quant_type == "nf4"
    assert cfg.qlora.compute_dtype == "bfloat16"


def test_schema_rejects_bogus_quant_type() -> None:
    """quant_type must be one of the literal values."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QLoRAConfig(quant_type="bogus")


def test_schema_rejects_bogus_compute_dtype() -> None:
    """compute_dtype must be one of the literal values."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QLoRAConfig(compute_dtype="float32")


def test_import_does_not_require_bitsandbytes() -> None:
    """qlora.py must not import bitsandbytes at module scope (lazy-import contract).

    Verified via AST inspection rather than monkeypatched re-import: re-importing
    the module triggers the @register("peft", "qlora") decorator a second time,
    which the registry rejects as a duplicate. AST inspection is more direct
    anyway — it pins the static contract.
    """
    import ast

    import esam3.peft_adapters.qlora as qlora_module

    # The module is already importable (this very import succeeded) — that
    # alone proves importing qlora does not require bitsandbytes at module
    # scope, since bitsandbytes is not installed in the CPU test environment.
    # The AST check below pins this as a static property of the source file,
    # not just a runtime artifact of the current sys.path.
    src = Path(qlora_module.__file__).read_text()
    tree = ast.parse(src)
    for node in tree.body:  # tree.body = top-level only; nested imports ignored
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "bitsandbytes", (
                    "qlora.py must not import bitsandbytes at module scope; "
                    "use a lazy import inside apply_qlora/load_qlora instead."
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "bitsandbytes", (
                "qlora.py must not `from bitsandbytes import ...` at module "
                "scope; use a lazy import inside apply_qlora/load_qlora."
            )

    # apply_qlora is reachable now; save_qlora/load_qlora land in Task 3.
    assert hasattr(qlora_module, "apply_qlora")


def test_apply_qlora_raises_helpful_importerror_when_bnb_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling apply_qlora without bnb raises ImportError naming the [qlora] extra.

    Setting sys.modules["bitsandbytes"] = None makes a fresh `import
    bitsandbytes` inside apply_qlora's lazy-import helper fail with
    ImportError. We do NOT evict esam3.peft_adapters.qlora from sys.modules
    here — re-importing it would re-fire the @register("peft", "qlora")
    decorator, which the registry rejects as a duplicate.
    """
    from esam3.peft_adapters.qlora import apply_qlora

    monkeypatch.setitem(sys.modules, "bitsandbytes", None)

    w = make_stub_wrapper()
    cfg = PEFTConfig(method="qlora")
    with pytest.raises(ImportError, match=r"\[qlora\]"):
        apply_qlora(w, cfg)


def test_save_qlora_raises_when_no_peft_model(tmp_path: Path) -> None:
    """save_qlora requires apply_qlora to have run first."""
    from esam3.peft_adapters.qlora import save_qlora

    w = make_stub_wrapper()
    assert w.peft_model is None
    with pytest.raises(RuntimeError, match="no PeftModel"):
        save_qlora(w, tmp_path)


def test_load_qlora_raises_when_peft_model_already_set(tmp_path: Path) -> None:
    """load_qlora refuses to overwrite a wrapper that already has an adapter."""
    from esam3.peft_adapters.qlora import load_qlora

    w = make_stub_wrapper()
    # Fake a previously-applied state. The real type is PeftModel; for this
    # guard test any non-None object suffices.
    w.peft_model = object()
    with pytest.raises(RuntimeError, match="already has a PeftModel"):
        load_qlora(w, tmp_path)


# ---------------------------------------------------------------------------
# CPU unit tests for _infer_quant_type_from_wrapper fallback chain
#
# These tests do NOT import bitsandbytes; they use a lightweight fake that
# mimics the `Linear4bit` / `Params4bit` shape just enough to exercise the
# attribute-read fallbacks in `_infer_quant_type_from_wrapper`.
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_bnb(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a tiny fake `bitsandbytes` module exposing `bnb.nn.Linear4bit`."""
    fake = types.ModuleType("bitsandbytes")
    fake_nn = types.ModuleType("bitsandbytes.nn")

    class _FakeLinear4bit(nn.Module):
        def __init__(
            self,
            *,
            weight_quant_type: str | None = None,
            module_quant_type: str | None = None,
        ) -> None:
            super().__init__()
            # Mimic a Params4bit weight with `.quant_type` directly on the weight.
            weight = nn.Parameter(nn.functional.normalize(nn.Linear(2, 2).weight))
            if weight_quant_type is not None:
                weight.quant_type = weight_quant_type  # type: ignore[attr-defined]
            self.weight = weight
            if module_quant_type is not None:
                self.quant_type = module_quant_type  # type: ignore[attr-defined]

    fake_nn.Linear4bit = _FakeLinear4bit  # type: ignore[attr-defined]
    fake.nn = fake_nn  # type: ignore[attr-defined]
    fake.__version__ = "0.fake.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", fake_nn)
    return fake


class _FakeWrapper:
    """Stand-in for Sam3Wrapper holding a `peft_model` attribute."""

    def __init__(self, peft_model: nn.Module) -> None:
        self.peft_model = peft_model


def test_infer_quant_type_primary_path(fake_bnb: types.ModuleType) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit(weight_quant_type="nf4")  # type: ignore[attr-defined]
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    assert _infer_quant_type_from_wrapper(wrapper) == "nf4"


def test_infer_quant_type_legacy_fallback(fake_bnb: types.ModuleType) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit(module_quant_type="fp4")  # type: ignore[attr-defined]
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    assert _infer_quant_type_from_wrapper(wrapper) == "fp4"


def test_infer_quant_type_raises_when_both_paths_missing(
    fake_bnb: types.ModuleType,
) -> None:
    from esam3.peft_adapters.qlora import _infer_quant_type_from_wrapper

    fake_linear4bit = fake_bnb.nn.Linear4bit()  # no quant_type set anywhere
    model = nn.Sequential(fake_linear4bit)
    wrapper: Any = _FakeWrapper(model)
    with pytest.raises(RuntimeError, match="could not infer quant_type"):
        _infer_quant_type_from_wrapper(wrapper)


def test_has_plain_nn_linear_ignores_lora_adapter_children() -> None:
    """The tightened predicate must ignore lora_A/lora_B nn.Linears but flag base leaks."""
    from tests.helpers.lora_predicates import has_plain_nn_linear as _has_plain_nn_linear

    # Fake LoRA adapter wrapper: holds a Linear4bit-shape sentinel as base, plus
    # full-precision lora_A / lora_B nn.Linear adapters (mimicking peft.tuners.lora.bnb.Linear4bit).
    class _Linear4bitSentinel(nn.Linear):
        """Subclass of nn.Linear (mimics bnb.nn.Linear4bit subclassing)."""

    class _FakeLoraWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = _Linear4bitSentinel(4, 4)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})

    # Case 1: all base Linears already swapped (only Linear4bitSentinel + lora adapters).
    # Expected: predicate returns False.
    clean = nn.Sequential(_FakeLoraWrapper(), _FakeLoraWrapper())
    assert not _has_plain_nn_linear(clean), (
        "predicate must not flag lora_A/lora_B adapter Linears as base leaks"
    )

    # Case 2: introduce a real base-Linear leak alongside the LoRA-wrapped layers.
    # Expected: predicate returns True (the leaked plain nn.Linear is NOT under a lora_* path).
    leaked = nn.Sequential(_FakeLoraWrapper(), nn.Linear(4, 4))
    assert _has_plain_nn_linear(leaked), "predicate must still flag a true base nn.Linear leak"


# ---------------------------------------------------------------------------
# Regression: _collect_linear_names must exclude children of
# nn.MultiheadAttention so that out_proj is NOT replaced by Linear4bit.
# Background: nn.MultiheadAttention.forward delegates to
# F.multi_head_attention_forward, which calls F.linear directly on
# mha.out_proj.weight (raw uint8 4-bit storage if quantized), bypassing the
# Linear4bit module dispatch and raising
# `RuntimeError: self and mat2 must have the same dtype, but got Float and
# Byte` on the first QLoRA forward through sam3's decoder.
# ---------------------------------------------------------------------------


def test_collect_linear_names_excludes_mha_children() -> None:
    """_collect_linear_names must NOT include any nn.Linear under nn.MultiheadAttention."""
    from esam3.peft_adapters.qlora import _collect_linear_names

    class _DecoderLikeBlock(nn.Module):
        """Mirrors sam3's decoder.py layer shape: MHA + standalone FFN Linears."""

        def __init__(self) -> None:
            super().__init__()
            # Native MHA — its `out_proj` (and any other internal Linears) MUST be skipped.
            self.self_attn = nn.MultiheadAttention(embed_dim=8, num_heads=2)
            self.ca_text = nn.MultiheadAttention(embed_dim=8, num_heads=2)
            # FFN — standalone Linears that SHOULD be picked up.
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)

    base = _DecoderLikeBlock()
    names = _collect_linear_names(base)

    # FFN Linears must be present.
    assert "fc1" in names
    assert "fc2" in names

    # No name under any nn.MultiheadAttention may appear.
    assert not any(n.startswith("self_attn") for n in names), (
        f"self_attn descendants leaked into quantization set: {names}"
    )
    assert not any(n.startswith("ca_text") for n in names), (
        f"ca_text descendants leaked into quantization set: {names}"
    )

    # Specifically, out_proj (the historical failure mode) must be absent.
    assert "self_attn.out_proj" not in names
    assert "ca_text.out_proj" not in names


def test_collect_linear_names_keeps_all_linears_when_no_mha() -> None:
    """Pin behavior on a model with no nn.MultiheadAttention: every Linear is collected."""
    from esam3.peft_adapters.qlora import _collect_linear_names

    class _ViTLikeBlock(nn.Module):
        """Mirrors sam3's ViT-Det block shape: custom q/k/v as bare Linears, FFN Linears."""

        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(8, 24)  # combined q,k,v projection (single Linear)
            self.proj = nn.Linear(8, 8)
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)

    base = _ViTLikeBlock()
    names = _collect_linear_names(base)
    assert set(names) == {"qkv", "proj", "fc1", "fc2"}


def test_collect_linear_names_handles_nested_mha() -> None:
    """A model with MHA nested inside a container still excludes the MHA's children."""
    from esam3.peft_adapters.qlora import _collect_linear_names

    class _Container(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(embed_dim=8, num_heads=2),
                    nn.MultiheadAttention(embed_dim=8, num_heads=2),
                ]
            )
            self.head = nn.Linear(8, 4)

    base = _Container()
    names = _collect_linear_names(base)
    # Only the top-level standalone Linear should appear; nothing under `layers.*`.
    assert names == ["head"], f"expected only 'head'; got {names}"


# ---------------------------------------------------------------------------
# Second-pass MHA exclusion: sam3.model.model_misc.MultiheadAttention is a
# DIFFERENT class from torch.nn.MultiheadAttention but has the same
# anti-pattern (multi_head_attention_forward extracts out_proj.weight as a
# raw tensor and passes it to F.linear). Both must be excluded.
# ---------------------------------------------------------------------------


def test_mha_exclusion_includes_sam3_custom_mha(monkeypatch: pytest.MonkeyPatch) -> None:
    """_mha_exclusion_types must include sam3's custom MultiheadAttention when sam3 is importable.

    We patch sys.modules with a fake sam3.model.model_misc that exposes a
    sentinel MultiheadAttention class, then re-execute the import path to
    confirm the exclusion picks it up.
    """
    from esam3.peft_adapters.qlora import _mha_exclusion_types

    # The real sam3 may or may not be installed in this test env. To assert
    # the contract regardless, install a fake sam3.model.model_misc with a
    # known sentinel class and confirm it ends up in the exclusion tuple.
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3_model = types.ModuleType("sam3.model")
    fake_sam3_model_misc = types.ModuleType("sam3.model.model_misc")

    class _Sam3MHA_Sentinel(nn.Module):
        pass

    fake_sam3_model_misc.MultiheadAttention = _Sam3MHA_Sentinel
    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model", fake_sam3_model)
    monkeypatch.setitem(sys.modules, "sam3.model.model_misc", fake_sam3_model_misc)

    excluded = _mha_exclusion_types()
    assert nn.MultiheadAttention in excluded, "torch MHA must always be excluded"
    assert _Sam3MHA_Sentinel in excluded, (
        f"sam3.model.model_misc.MultiheadAttention must be excluded; got {excluded}"
    )


def test_mha_exclusion_degrades_to_torch_only_without_sam3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_mha_exclusion_types still works when sam3.model.model_misc is missing."""
    from esam3.peft_adapters.qlora import _mha_exclusion_types

    # Force the import path to fail by setting a sentinel that raises.
    monkeypatch.setitem(sys.modules, "sam3.model.model_misc", None)

    excluded = _mha_exclusion_types()
    assert excluded == (nn.MultiheadAttention,), (
        f"without sam3, only torch MHA should be excluded; got {excluded}"
    )


def test_collect_linear_names_excludes_sam3_custom_mha_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_collect_linear_names must skip Linears under sam3's custom MultiheadAttention.

    Pins the actual failure mode that broke the QLoRA release-tier test on
    the manual GPU pass (issue #44): sam3's custom MultiheadAttention class
    holds an out_proj nn.Linear and bypasses module dispatch the same way
    torch's built-in MHA does.
    """
    from esam3.peft_adapters.qlora import _collect_linear_names

    # Fake-install a sam3.model.model_misc module exposing a custom MHA class.
    # We mirror the real shape: an nn.Module with an out_proj nn.Linear child
    # (sam3's class also has in_proj_weight as a raw Parameter, but only
    # out_proj is an nn.Linear that _collect_linear_names would sweep up).
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3_model = types.ModuleType("sam3.model")
    fake_sam3_model_misc = types.ModuleType("sam3.model.model_misc")

    class _Sam3CustomMHA(nn.Module):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.out_proj = nn.Linear(embed_dim, embed_dim)

    fake_sam3_model_misc.MultiheadAttention = _Sam3CustomMHA
    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model", fake_sam3_model)
    monkeypatch.setitem(sys.modules, "sam3.model.model_misc", fake_sam3_model_misc)

    class _DecoderLikeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cross_attend_prompt = _Sam3CustomMHA(embed_dim=8)  # sam3 custom MHA
            self.self_attn = nn.MultiheadAttention(embed_dim=8, num_heads=2)  # torch MHA
            self.fc1 = nn.Linear(8, 16)  # bare Linear — should be picked up
            self.fc2 = nn.Linear(16, 8)  # bare Linear — should be picked up

    base = _DecoderLikeBlock()
    names = _collect_linear_names(base)

    # Both MHA classes' out_proj must be skipped.
    assert "cross_attend_prompt.out_proj" not in names, (
        f"sam3 custom MHA's out_proj leaked into quantization set: {names}"
    )
    assert "self_attn.out_proj" not in names, (
        f"torch MHA's out_proj leaked into quantization set: {names}"
    )

    # Bare Linears unaffected.
    assert "fc1" in names
    assert "fc2" in names

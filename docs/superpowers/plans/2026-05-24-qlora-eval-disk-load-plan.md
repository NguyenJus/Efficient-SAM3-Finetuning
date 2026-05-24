# QLoRA Eval Disk-Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-24-qlora-eval-disk-load-design.md`](../specs/2026-05-24-qlora-eval-disk-load-design.md)
**Issue:** [#98](https://github.com/NguyenJus/custom-sam-peft/issues/98) — labeled `hardening-followup`
**Branch:** `worktree-qlora-eval-disk-load-98`

**Goal:** Wire `run_eval` to load QLoRA checkpoints from disk via a new `PEFTMethod.load_from_disk` protocol method, and restore the channel adapter in the same path.

**Architecture:** Two source files change (`peft_adapters/__init__.py` and `eval/runner.py`). The new `load_from_disk` protocol method delegates lazily to the already-tested `load_lora` / `load_qlora` functions, preserving the bitsandbytes isolation contract. The `_load_channel_adapter` helper (already used in predict and train-resume paths) is wired into the eval disk-load path. No new core machinery is introduced — this is entirely a wiring change.

**Tech Stack:** Python 3.12, pytest (CPU-only mocks for all new unit tests; GPU deferred), ruff, existing `load_lora`, `load_qlora`, `_load_channel_adapter` helpers.

---

## Drift notes (planner-verified against branch HEAD)

- **`peft_adapters/__init__.py`** is at the post-#100 state (registry-driven `make_peft_method`, `@register` decorators on `LoraAdapter`/`QloraAdapter`, `cast`/`RegistryError`/`lookup`/`register` already imported). Spec line references for §3.1 are accurate.
- **`typing` import (line 21):** currently `from typing import Protocol, cast, runtime_checkable` — `Any` is absent and must be added.
- **`eval/runner.py` line 23:** `from custom_sam_peft.peft_adapters.lora import load_lora` is present and must be removed.
- **`eval/runner.py` lines 106–110:** the `ValueError` guard is present and must be deleted.
- **`eval/runner.py` line 133:** `load_lora(wrapper, resolved_checkpoint)` is present and must be replaced.
- **`test_eval_runner.py` six `load_lora` patches:** confirmed at lines 63, 101, 159, 230, 290, 328. All six must be repointed in the same commit that removes the `load_lora` import from `runner.py`.
- **`test_peft_method_protocol.py` line 106–107:** `test_qlora_adapter_supports_checkpoint_load_from_disk_false` asserts `is False`; must be flipped to `is True` and renamed.
- **`test_cli.py` lines 192–215:** `test_eval_command_rejects_qlora_method` must be renamed and flipped per spec §4.4.
- **GPU test:** `test_save_load_qlora_roundtrip` (lines 104–128 of `tests/integration/test_peft_qlora_real.py`) ends immediately after the LoRA param parity assertion. Forward-output capture must be inserted **before** `del w1` (currently line 118). This environment has no compatible GPU — write the code; do not run it.

---

## File Map

### Modified files

```
src/custom_sam_peft/peft_adapters/__init__.py     # Add Any import; add load_from_disk to Protocol;
                                                   # flip QloraAdapter.supports_checkpoint_load_from_disk;
                                                   # add LoraAdapter.load_from_disk;
                                                   # add QloraAdapter.load_from_disk;
                                                   # update module docstring (Task 1)

src/custom_sam_peft/eval/runner.py                # Remove load_lora import; add _load_channel_adapter import;
                                                   # delete ValueError guard; replace load_lora call with
                                                   # _peft_method.load_from_disk + _load_channel_adapter;
                                                   # update run_eval docstring (Task 1)

tests/unit/test_peft_method_protocol.py           # Flip/rename qlora supports_checkpoint test;
                                                   # add load_from_disk protocol structure test;
                                                   # add LoraAdapter delegation test;
                                                   # add QloraAdapter delegation test (Task 2)

tests/unit/test_eval_runner.py                    # Flip/rename qlora-reject test;
                                                   # add _load_channel_adapter assertion for lora path;
                                                   # repoint all 6 load_lora patches (Task 2)

tests/unit/test_cli.py                            # Flip/rename qlora CLI rejection test (Task 2)

tests/integration/test_peft_qlora_real.py         # Extend roundtrip test with forward-output parity
                                                   # assertion (write-only; GPU deferred) (Task 2)
```

### No new files

All changes are edits to existing files.

---

## Routing summary (for orchestrator)

| Task | Suggested impl model | Reviewer | Notes |
| --- | --- | --- | --- |
| 0 | n/a | n/a | Pre-flight; orchestrator runs commands directly. |
| 1 | sonnet/high | sonnet/high | Source-code wiring; two files; no new machinery. |
| 2 | sonnet/high | sonnet/high | Test edits; six-patch repoint hazard is explicit. |
| 3 | n/a | n/a | Manual gate + PR; orchestrator runs commands directly. |

**Parallelization:** Task 1 (source) and Task 2 (tests) must be **sequential** — the six patch repoints in Task 2 depend on the import removal in Task 1.

---

## Task 0: Verify clean baseline

**Files:** none (commands only)

This task is a pre-flight gate. Orchestrator runs these directly.

- [ ] **Step 0a: Confirm working tree is clean and on the right branch**

```bash
git status
git branch --show-current
```

Expected: branch `qlora-eval-disk-load-98`, working tree clean (only spec + plan files if not yet committed). If dirty, halt.

- [ ] **Step 0b: Confirm venv has dev extras**

```bash
uv sync --extra dev
```

Expected: resolves without errors. Required once per fresh worktree; subsequent runs are fast.

- [ ] **Step 0c: Confirm baseline unit tests pass**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green. If anything is red before changes, halt and surface — Task 1 cannot validate against a broken baseline.

- [ ] **Step 0d: Confirm ruff is clean**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: both clean.

---

## Task 1: Source-code wiring — `peft_adapters/__init__.py` and `eval/runner.py`

**Files:**
- Modify: `src/custom_sam_peft/peft_adapters/__init__.py` (lines 21, 63–70, 92–94, 115–116)
- Modify: `src/custom_sam_peft/eval/runner.py` (lines 23, 82–91, 106–110, 129–133)

**Objective:** Add `load_from_disk` to the `PEFTMethod` Protocol and both adapter implementations; flip `QloraAdapter.supports_checkpoint_load_from_disk` to `True`; update the eval runner to dispatch through the protocol and restore the channel adapter.

---

### 1A — `peft_adapters/__init__.py` changes

- [ ] **Step 1A-1: Add `Any` to the typing import (line 21)**

Current line 21:
```python
from typing import Protocol, cast, runtime_checkable
```

Replace with:
```python
from typing import Any, Protocol, cast, runtime_checkable
```

- [ ] **Step 1A-2: Add `load_from_disk` to the `PEFTMethod` Protocol (after line 70, before the closing of the class)**

Current end of the `PEFTMethod` Protocol (lines 63–70):
```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        """Return True if this method can load a checkpoint from disk without
        a pre-loaded model wrapper.

        LoRA returns True. QLoRA returns False (requires a live wrapper with
        quantized base; disk-only load is deferred to a follow-up PR).
        """
        ...
```

Replace the entire `supports_checkpoint_load_from_disk` Protocol method (updating the docstring to remove the deferred-PR language, per spec §3.1 Change B) AND append `load_from_disk` after it:

```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        """Return True if this method can load a checkpoint from disk without
        a pre-loaded model wrapper.

        LoRA returns True. QLoRA returns True (load_qlora reconstructs the 4-bit
        quantized base from saved custom_sam_peft_qlora.json metadata, then loads
        the LoRA adapter weights via PeftModel.from_pretrained).
        """
        ...

    def load_from_disk(self, wrapper: Any, dirpath: Any) -> Any:
        """Load a checkpoint from disk into a freshly-built wrapper.

        Rebuilds the PEFT-adapted model from the saved checkpoint directory
        (``dirpath``), mutating ``wrapper`` in place. Returns ``wrapper``.

        LoRA implementation delegates to ``load_lora(wrapper, dirpath)``.
        QLoRA implementation delegates to ``load_qlora(wrapper, dirpath)``,
        which reconstructs the 4-bit quantized base from saved metadata before
        loading the LoRA adapter weights.

        Both implementations import their respective loaders lazily inside the
        method body so that LoRA-only users never import bitsandbytes.
        """
        ...
```

- [ ] **Step 1A-3: Flip `QloraAdapter.supports_checkpoint_load_from_disk` to return `True` (line 115–116)**

Current (lines 115–116):
```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        return False
```

Replace with:
```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        return True
```

- [ ] **Step 1A-4: Add `LoraAdapter.load_from_disk` after `LoraAdapter.supports_checkpoint_load_from_disk` (after line 93)**

Current end of `LoraAdapter` class (line 92–93):
```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        return True
```

Insert after that method (still inside the `LoraAdapter` class, before the blank line and the `@register("peft_method", "qlora")` decorator):

```python
    def load_from_disk(self, wrapper: Any, dirpath: Any) -> Any:
        from custom_sam_peft.peft_adapters.lora import load_lora

        return load_lora(wrapper, dirpath)
```

- [ ] **Step 1A-5: Add `QloraAdapter.load_from_disk` after `QloraAdapter.supports_checkpoint_load_from_disk` (after line 116)**

Current end of `QloraAdapter` class (line 115–116):
```python
    def supports_checkpoint_load_from_disk(self) -> bool:
        return True  # (after step 1A-3)
```

Insert after that method (still inside the `QloraAdapter` class, before the blank line and `def method_pretty_name`):

```python
    def load_from_disk(self, wrapper: Any, dirpath: Any) -> Any:
        from custom_sam_peft.peft_adapters.qlora import load_qlora

        return load_qlora(wrapper, dirpath)
```

- [ ] **Step 1A-6: Update module docstring (lines 1–16) to document `load_from_disk`**

Append the following line inside the existing docstring, after the registered factories block (after the `lookup("peft_method", "qlora")` line, before the final `For method-dispatch...` paragraph):

```python
"""PEFT adapter package.

Documented seam: trainers, evaluators, and checkpoint code interact with
PEFT adapters through the ``PEFTMethod`` protocol below. They must not
branch on ``cfg.peft.method`` strings.

Registered factories:
  ``lookup("peft", "lora")``         → ``apply_lora``         (wrapper, cfg) → Sam3Wrapper
  ``lookup("peft", "qlora")``        → ``apply_qlora``        (wrapper, cfg) → Sam3Wrapper
  ``lookup("peft_method", "lora")``  → ``LoraAdapter``        () → PEFTMethod
  ``lookup("peft_method", "qlora")`` → ``QloraAdapter``       () → PEFTMethod

For disk-load dispatch in evaluators/CLI:
  ``_peft_method.load_from_disk(wrapper, dirpath)``  → delegates to load_lora or load_qlora

For method-dispatch decisions (optimizer, autocast, checkpoint detection)
call the appropriate ``LoraAdapter`` or ``QloraAdapter`` instance methods
instead of testing ``cfg.peft.method``.
"""
```

- [ ] **Step 1A-7: Smoke-check `peft_adapters/__init__.py` imports cleanly with no bitsandbytes at module level**

```bash
uv run python -c "from custom_sam_peft.peft_adapters import PEFTMethod, LoraAdapter, QloraAdapter, make_peft_method; print('OK')"
grep "bitsandbytes" src/custom_sam_peft/peft_adapters/__init__.py
grep "load_lora\|load_qlora" src/custom_sam_peft/peft_adapters/__init__.py
```

Expected: `OK` printed; `grep bitsandbytes` returns no matches; `grep load_lora\|load_qlora` returns no matches (both are lazy inside method bodies, not module-level).

---

### 1B — `eval/runner.py` changes

- [ ] **Step 1B-1: Remove the `load_lora` import and add `_load_channel_adapter` import (line 23)**

Current line 23:
```python
from custom_sam_peft.peft_adapters.lora import load_lora
```

Replace with:
```python
from custom_sam_peft.train.checkpoint import _load_channel_adapter
```

- [ ] **Step 1B-2: Delete the `ValueError` guard for non-LoRA PEFT methods (lines 106–110)**

Current lines 106–110:
```python
    _peft_method = make_peft_method(resolved_peft_method)
    if model is None and not _peft_method.supports_checkpoint_load_from_disk():
        raise ValueError(
            f"checkpoint loading currently supports only LoRA adapters; "
            f"got peft.method={resolved_peft_method!r}"
        )
```

Replace with:
```python
    _peft_method = make_peft_method(resolved_peft_method)
```

- [ ] **Step 1B-3: Replace `load_lora(wrapper, resolved_checkpoint)` with protocol dispatch + channel adapter restore (line 133)**

Current lines 129–133:
```python
    if model is None:
        wrapper = load_sam31(
            cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
        )
        load_lora(wrapper, resolved_checkpoint)
```

Replace with:
```python
    if model is None:
        wrapper = load_sam31(
            cfg.model, channels=cfg.data.channels, channel_semantics=cfg.data.channel_semantics
        )
        _peft_method.load_from_disk(wrapper, resolved_checkpoint)
        _load_channel_adapter(wrapper, resolved_checkpoint)
```

- [ ] **Step 1B-4: Update the `run_eval` docstring — remove the QLoRA-rejection `Raises` entry (lines 82–91)**

Current `Raises` block in the `run_eval` docstring (lines 87–91):
```python
    Raises:
        ValueError: cfg.peft.method != 'lora' AND model is None (QLoRA load
            from disk is not yet supported; pre-loaded wrappers bypass this).
        ValueError: split == 'test' and cfg.data.test is None.
        ValueError: neither ``checkpoint`` nor ``artifacts`` provided.
```

Replace with (remove only the QLoRA-rejection entry):
```python
    Raises:
        ValueError: split == 'test' and cfg.data.test is None.
        ValueError: neither ``checkpoint`` nor ``artifacts`` provided.
```

Also update the docstring body paragraph that still references `load_lora` by name. Current line 82:
```python
      - ``model``: pre-loaded + adapted wrapper; skips ``load_sam31`` + ``load_lora``.
```

Replace with:
```python
      - ``model``: pre-loaded + adapted wrapper; skips ``load_sam31`` + adapter load.
```

- [ ] **Step 1B-5: Run ruff on both changed source files**

```bash
uv run ruff check src/custom_sam_peft/peft_adapters/__init__.py src/custom_sam_peft/eval/runner.py
uv run ruff format src/custom_sam_peft/peft_adapters/__init__.py src/custom_sam_peft/eval/runner.py
```

Expected: both clean. Fix any lint errors before proceeding.

- [ ] **Step 1B-6: Confirm `runner.py` contains no `.method ==` literals (protocol-seam guard)**

```bash
grep -n "\.method ==" src/custom_sam_peft/eval/runner.py
```

Expected: zero matches. This is also verified by the existing `test_eval_runner_does_not_branch_on_method_name` test.

- [ ] **Step 1B-7: Confirm no top-level `load_lora` reference in `runner.py`**

```bash
grep -n "load_lora" src/custom_sam_peft/eval/runner.py
```

Expected: zero matches.

- [ ] **Step 1B-8: Commit source changes (Commit 1 of 2)**

```bash
git add src/custom_sam_peft/peft_adapters/__init__.py src/custom_sam_peft/eval/runner.py
git commit -m "feat(peft): add load_from_disk protocol method; wire QLoRA eval disk-load + channel adapter (#98)"
```

**Acceptance for this commit:**
- `QloraAdapter().supports_checkpoint_load_from_disk()` returns `True`.
- `hasattr(LoraAdapter(), 'load_from_disk')` and `hasattr(QloraAdapter(), 'load_from_disk')` are both `True`.
- `grep "bitsandbytes" src/custom_sam_peft/peft_adapters/__init__.py` — no matches.
- `grep "load_lora\|load_qlora" src/custom_sam_peft/peft_adapters/__init__.py` — no module-level matches (only inside method bodies).
- `grep "load_lora" src/custom_sam_peft/eval/runner.py` — no matches.
- `grep "checkpoint loading currently supports only LoRA" src/custom_sam_peft/eval/runner.py` — no matches.

---

## Task 2: Tests — flip blocking tests, add new assertions, repoint six patches

**Files:**
- Modify: `tests/unit/test_peft_method_protocol.py` (lines 49–50, 106–107, and new tests after lines 87, 122)
- Modify: `tests/unit/test_eval_runner.py` (lines 35–38, 63, 101, 159, 230, 290, 328, and new test)
- Modify: `tests/unit/test_cli.py` (lines 192–215)
- Modify: `tests/integration/test_peft_qlora_real.py` (lines 111–128; forward-output extension)

**Objective:** Flip the three tests that previously asserted the now-removed rejection behavior; add delegation tests for `load_from_disk`; add `_load_channel_adapter` assertions for both LoRA and QLoRA paths; and extend the GPU round-trip test with a forward-output parity check (code written; GPU verification deferred).

**CRITICAL ordering hazard:** Steps 2A and 2B must land in the **same commit** (Commit 2). Removing the `load_lora` import from `runner.py` (already done in Task 1) means `custom_sam_peft.eval.runner.load_lora` no longer exists as an attribute. The six `monkeypatch.setattr("custom_sam_peft.eval.runner.load_lora", ...)` calls at lines 63, 101, 159, 230, 290, 328 of `test_eval_runner.py` will raise `AttributeError` at setup time because `monkeypatch.setattr` uses `raising=True` by default. All six must be repointed in this task.

---

### 2A — `tests/unit/test_peft_method_protocol.py`

- [ ] **Step 2A-1: Write the failing `load_from_disk` protocol structure test and verify it currently fails**

At line 49–50, after `test_peft_method_protocol_declares_supports_checkpoint_load_from_disk`, add:

```python
def test_peft_method_protocol_declares_load_from_disk() -> None:
    assert hasattr(PEFTMethod, "load_from_disk")
```

Run to confirm it passes (Task 1 already added the method to the Protocol):

```bash
uv run pytest tests/unit/test_peft_method_protocol.py::test_peft_method_protocol_declares_load_from_disk -v --no-cov
```

Expected: PASS (Task 1 already added `load_from_disk` to the Protocol).

- [ ] **Step 2A-2: Rename and flip `test_qlora_adapter_supports_checkpoint_load_from_disk_false` (line 106–107)**

Current lines 106–107:
```python
def test_qlora_adapter_supports_checkpoint_load_from_disk_false() -> None:
    assert QloraAdapter().supports_checkpoint_load_from_disk() is False
```

Replace with:
```python
def test_qlora_adapter_supports_checkpoint_load_from_disk_true() -> None:
    assert QloraAdapter().supports_checkpoint_load_from_disk() is True
```

Run to verify it passes:

```bash
uv run pytest tests/unit/test_peft_method_protocol.py::test_qlora_adapter_supports_checkpoint_load_from_disk_true -v --no-cov
```

Expected: PASS.

- [ ] **Step 2A-3: Add `LoraAdapter.load_from_disk` delegation test (after line 87, in the `# 2. LoraAdapter` section)**

Insert after `test_lora_adapter_detect_method_raises_on_qlora_marker` (line 87):

```python
def test_lora_adapter_load_from_disk_delegates_to_load_lora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LoraAdapter.load_from_disk must call load_lora with (wrapper, dirpath) and return its result."""
    from unittest.mock import MagicMock

    fake_wrapper = MagicMock()
    sentinel = MagicMock()

    monkeypatch.setattr(
        "custom_sam_peft.peft_adapters.lora.load_lora",
        lambda w, d: (sentinel if w is fake_wrapper and d == tmp_path else None),
    )

    result = LoraAdapter().load_from_disk(fake_wrapper, tmp_path)
    assert result is sentinel
```

Note: `LoraAdapter.load_from_disk` uses a lazy `from custom_sam_peft.peft_adapters.lora import load_lora`. Patching the module-level attribute `custom_sam_peft.peft_adapters.lora.load_lora` is the correct target — the lazy import resolves to this name at call time.

Run to verify it passes:

```bash
uv run pytest "tests/unit/test_peft_method_protocol.py::test_lora_adapter_load_from_disk_delegates_to_load_lora" -v --no-cov
```

Expected: PASS.

- [ ] **Step 2A-4: Add `QloraAdapter.load_from_disk` delegation test (after line 122, in the `# 3. QloraAdapter` section)**

Insert after `test_qlora_adapter_detect_method_raises_without_meta_file` (line 122):

```python
def test_qlora_adapter_load_from_disk_delegates_to_load_qlora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QloraAdapter.load_from_disk must call load_qlora with (wrapper, dirpath) and return its result."""
    from unittest.mock import MagicMock

    fake_wrapper = MagicMock()
    sentinel = MagicMock()

    monkeypatch.setattr(
        "custom_sam_peft.peft_adapters.qlora.load_qlora",
        lambda w, d: (sentinel if w is fake_wrapper and d == tmp_path else None),
    )

    result = QloraAdapter().load_from_disk(fake_wrapper, tmp_path)
    assert result is sentinel
```

Note: same lazy-import rationale as step 2A-3 — patch `custom_sam_peft.peft_adapters.qlora.load_qlora`.

Run to verify it passes:

```bash
uv run pytest "tests/unit/test_peft_method_protocol.py::test_qlora_adapter_load_from_disk_delegates_to_load_qlora" -v --no-cov
```

Expected: PASS.

- [ ] **Step 2A-5: Run the full `test_peft_method_protocol.py` suite**

```bash
uv run pytest tests/unit/test_peft_method_protocol.py -v --no-cov
```

Expected: all pass, including the unchanged `test_eval_runner_does_not_branch_on_method_name` (runner no longer contains `.method ==`).

---

### 2B — `tests/unit/test_eval_runner.py`

**CRITICAL:** Steps 2B-1 through 2B-4 collectively perform the six-patch repoint. After Task 1 removed `load_lora` from `runner.py`, all six tests that patch `custom_sam_peft.eval.runner.load_lora` will error at collection time with `AttributeError`. All six must be fixed before the test suite can run.

The correct replacement target is `custom_sam_peft.peft_adapters.lora.load_lora`. Because `LoraAdapter.load_from_disk` lazy-imports `load_lora` from that module, patching the module-level name is honored at call time.

- [ ] **Step 2B-1: Flip and rename `test_run_eval_rejects_non_lora_peft` (lines 35–38)**

Current lines 35–38:
```python
def test_run_eval_rejects_non_lora_peft(tmp_path: Path) -> None:
    cfg = _make_cfg(peft_method="qlora")
    with pytest.raises(ValueError, match="lora"):
        run_eval(cfg, checkpoint=tmp_path, split="val")
```

Replace with:
```python
def test_run_eval_dispatches_qlora_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval with peft_method='qlora' and model=None must dispatch via load_from_disk
    (calling load_qlora) and call _load_channel_adapter, without raising."""
    cfg = _make_cfg(peft_method="qlora")

    qlora_loader_calls: list[tuple[object, object]] = []
    channel_adapter_calls: list[tuple[object, object]] = []

    def fake_load_qlora(wrapper: object, dirpath: object) -> object:
        qlora_loader_calls.append((wrapper, dirpath))
        return wrapper

    def fake_load_channel_adapter(wrapper: object, dirpath: object) -> None:
        channel_adapter_calls.append((wrapper, dirpath))

    monkeypatch.setattr(
        "custom_sam_peft.peft_adapters.qlora.load_qlora", fake_load_qlora
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner._load_channel_adapter", fake_load_channel_adapter
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    fake_report = MagicMock()
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    result = run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert result is fake_report
    assert len(qlora_loader_calls) == 1, "load_qlora must be called exactly once"
    assert len(channel_adapter_calls) == 1, "_load_channel_adapter must be called exactly once"
    # Verify dirpath is the resolved checkpoint.
    _, dirpath = qlora_loader_calls[0]
    assert dirpath == tmp_path
```

- [ ] **Step 2B-2: Add `test_run_eval_lora_calls_load_channel_adapter` (add after `test_run_eval_rejects_test_split_when_data_test_none`, which is at line 41–44)**

Insert the following new test after the `test_run_eval_rejects_test_split_when_data_test_none` function (after line 44):

```python
def test_run_eval_lora_calls_load_channel_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_eval with peft_method='lora' and model=None must call _load_channel_adapter."""
    cfg = _make_cfg(peft_method="lora")
    channel_adapter_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        "custom_sam_peft.peft_adapters.lora.load_lora",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner._load_channel_adapter",
        lambda wrapper, dirpath: channel_adapter_calls.append((wrapper, dirpath)),
    )
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.lookup",
        lambda *_a, **_kw: lambda *a, **kw: MagicMock(__len__=lambda self: 0, class_names=[]),
    )
    monkeypatch.setattr("custom_sam_peft.eval.runner.load_sam31", lambda _m, **_kw: MagicMock())
    fake_report = MagicMock()
    monkeypatch.setattr(
        "custom_sam_peft.eval.runner.Evaluator",
        lambda _cfg: MagicMock(evaluate_and_save=MagicMock(return_value=fake_report)),
    )

    run_eval(cfg, checkpoint=tmp_path, split="val", output_dir=tmp_path)

    assert len(channel_adapter_calls) == 1, "_load_channel_adapter must be called exactly once"
    _, dirpath = channel_adapter_calls[0]
    assert dirpath == tmp_path
```

- [ ] **Step 2B-3: Repoint the six `load_lora` patches — lines 63, 101, 159, 230, 290, 328**

Each of the six occurrences is:
```python
monkeypatch.setattr("custom_sam_peft.eval.runner.load_lora", lambda *_a, **_kw: None)
```

Replace every one with:
```python
monkeypatch.setattr("custom_sam_peft.peft_adapters.lora.load_lora", lambda *_a, **_kw: None)
```

This applies to all six lines. After the replacement, confirm:

```bash
grep -n "eval.runner.load_lora" tests/unit/test_eval_runner.py
```

Expected: zero matches.

```bash
grep -n "peft_adapters.lora.load_lora" tests/unit/test_eval_runner.py
```

Expected: six matches.

- [ ] **Step 2B-4: Run `test_eval_runner.py` in full**

```bash
uv run pytest tests/unit/test_eval_runner.py -v --no-cov
```

Expected: all green. In particular:
- `test_run_eval_dispatches_qlora_from_disk` — PASS (QLoRA no longer raises; `load_qlora` and `_load_channel_adapter` each called once).
- `test_run_eval_lora_calls_load_channel_adapter` — PASS (`_load_channel_adapter` called once on LoRA path).
- `test_run_eval_accepts_prebuilt_val_dataset_and_model` — PASS (`_load_channel_adapter` not called when `model` is pre-loaded, because the `if model is None:` guard remains).
- All six previously patched tests — PASS (patches now target `custom_sam_peft.peft_adapters.lora.load_lora`).

---

### 2C — `tests/unit/test_cli.py`

- [ ] **Step 2C-1: Rename and flip `test_eval_command_rejects_qlora_method` (lines 192–215)**

Current lines 192–215:
```python
def test_eval_command_rejects_qlora_method(tmp_path: Path) -> None:
    """custom_sam_peft eval --checkpoint errors when peft.method is not lora."""
    from custom_sam_peft.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run: {name: t, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
  prompt_mode: text
peft: {method: qlora}
train: {epochs: 1}
"""
    )
    local_runner = CliRunner()
    result = local_runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--checkpoint", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "qlora" in _plain(result.output).lower() or "only lora" in _plain(result.output).lower()
```

Replace with:
```python
def test_eval_command_accepts_qlora_method(tmp_path: Path) -> None:
    """custom_sam_peft eval --checkpoint no longer rejects peft.method=qlora.

    The command will fail for other reasons (no real checkpoint on disk), but the
    failure must NOT be the old 'only LoRA adapters' guard. QLoRA is now accepted
    and dispatched via QloraAdapter.load_from_disk.
    """
    from custom_sam_peft.cli.main import app

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run: {name: t, output_dir: ./runs, seed: 0}
data:
  format: coco
  train: {annotations: t.json, images: t/}
  val: {annotations: v.json, images: v/}
  prompt_mode: text
peft: {method: qlora}
train: {epochs: 1}
"""
    )
    local_runner = CliRunner()
    result = local_runner.invoke(
        app,
        ["eval", "--config", str(cfg_path), "--checkpoint", str(tmp_path)],
    )
    # Must NOT contain the old rejection message.
    assert "checkpoint loading currently supports only LoRA" not in _plain(result.output)
    assert "only lora" not in _plain(result.output).lower()
```

- [ ] **Step 2C-2: Run the renamed CLI test**

```bash
uv run pytest tests/unit/test_cli.py::test_eval_command_accepts_qlora_method -v --no-cov
```

Expected: PASS. The CLI invokes `run_eval`, which now dispatches through `QloraAdapter.load_from_disk` (which lazy-imports `load_qlora` and tries to open `custom_sam_peft_qlora.json`). It will fail with a file-not-found / checkpoint error — but NOT the old "only LoRA adapters" ValueError.

- [ ] **Step 2C-3: Run the full `test_cli.py` suite**

```bash
uv run pytest tests/unit/test_cli.py -v --no-cov
```

Expected: all green.

---

### 2D — `tests/integration/test_peft_qlora_real.py` (GPU deferred — write code only)

**IMPORTANT:** This environment has no compatible GPU (GTX 1080 / sm_61 requires `cu118` extras and the integration test requires `bitsandbytes` NF4 kernels). Write the code exactly as specified. Do NOT attempt to run it. Mark verification as deferred to the GPU tier (`gpu_local` marker). Do NOT claim this test passes.

- [ ] **Step 2D-1: Inspect `Sam3Wrapper.model.forward` signature before writing the dummy input**

Before writing the GPU extension code, inspect how `Evaluator` calls the wrapped model and what the actual forward signature is, so the dummy input shape is correct:

```bash
grep -n "def forward\|w\.model\|wrapper\.model" src/custom_sam_peft/eval/evaluator.py | head -20
grep -n "def forward" src/custom_sam_peft/models/sam3.py | head -10
```

The spec's `w1.model(_dummy_input)` with a `(1, 3, 1024, 1024)` raw tensor is **provisional**. SAM 3.1's forward may expect a dict of inputs rather than a raw tensor. The implementer must adapt the dummy input to match the actual `_Sam3ImageAdapter.forward` (or equivalent) call pattern that `Evaluator` uses. The tensor shape example in the spec is a starting point, not a guarantee.

- [ ] **Step 2D-2: Extend `test_save_load_qlora_roundtrip` with forward-output parity**

The existing test (lines 104–128) currently ends at:
```python
    for name, t1 in sd1.items():
        assert torch.allclose(t1.to(sd2[name].device), sd2[name], atol=0.0), f"mismatch on {name}"
```

**Insert before `del w1` (before the current line 118):**

```python
    # Capture a forward output from w1 before deleting it.
    # Use eval mode + fixed seed to suppress dropout and stochastic ops.
    # Input shape (1, C, H, W): SAM 3.1 canonical input is (1, 3, 1024, 1024).
    # ADAPT THIS: inspect Sam3Wrapper.model.forward (or the _Sam3ImageAdapter)
    # to confirm the actual call signature; the evaluator call pattern is the
    # canonical reference. If the model expects a dict of inputs rather than
    # a raw tensor, construct that dict here instead.
    import torch as _torch

    _torch.manual_seed(0)
    w1.model.eval()
    _dummy_input = _torch.zeros(1, 3, 1024, 1024, device="cuda", dtype=_torch.float32)
    with _torch.no_grad():
        _out_w1 = w1.model(_dummy_input)
    # Store only the output tensor on CPU; do NOT keep w1 alive.
    # If the model returns a dict/list, adapt to extract a representative scalar tensor,
    # e.g. _out_w1["pred_masks"][0].detach().cpu()
    _out_w1_cpu = _out_w1.detach().cpu() if isinstance(_out_w1, _torch.Tensor) else None
```

**Insert after the existing `load_qlora(w2, tmp_path)` call (after the current line 123):**

```python
    # Forward-output parity: w2 must produce the same output as w1 on the same input.
    if _out_w1_cpu is not None:
        _torch.manual_seed(0)
        w2.model.eval()
        _dummy_input2 = _torch.zeros(1, 3, 1024, 1024, device="cuda", dtype=_torch.float32)
        with _torch.no_grad():
            _out_w2 = w2.model(_dummy_input2)
        _out_w2_cpu = _out_w2.detach().cpu() if isinstance(_out_w2, _torch.Tensor) else None
        assert _out_w2_cpu is not None, "w2 forward output unexpectedly None"
        assert _torch.allclose(_out_w1_cpu, _out_w2_cpu, atol=1e-4, rtol=1e-4), (
            f"forward output mismatch after load_qlora roundtrip; "
            f"max abs diff={(_out_w1_cpu - _out_w2_cpu).abs().max().item():.6f}"
        )
```

**Tolerance note:** `atol=1e-4, rtol=1e-4` is appropriate for bfloat16/float16 after a quantize→save→dequant→load cycle. The LoRA weights have `atol=0.0` (exact), but 4-bit base re-quantization introduces small dequant rounding errors. Verify and adjust on the GTX 1080 (sm_61, float16 `compute_dtype`) before merging.

The existing `@pytest.mark.skipif(not _bnb_available(), ...)`, `pytestmark` markers (`requires_checkpoint`, `requires_compatible_gpu`, `gpu_local`), and `del w1 / gc.collect() / torch.cuda.empty_cache()` sequence are **unchanged**.

**GPU verification deferred:** This test cannot be run in this environment. It is marked `gpu_local` and will be verified at the GPU tier before the PR merges.

---

### 2E — Run full unit suite and ruff gate

- [ ] **Step 2E-1: Run the full non-GPU test suite**

```bash
uv run pytest tests/unit -x -q --no-cov
```

Expected: all green. The 80% coverage gate is checked separately in Step 3A below (requires the full non-GPU suite, not just `tests/unit`).

- [ ] **Step 2E-2: Run ruff on all modified test files**

```bash
uv run ruff check tests/unit/test_peft_method_protocol.py tests/unit/test_eval_runner.py tests/unit/test_cli.py tests/integration/test_peft_qlora_real.py
uv run ruff format tests/unit/test_peft_method_protocol.py tests/unit/test_eval_runner.py tests/unit/test_cli.py tests/integration/test_peft_qlora_real.py
```

Expected: clean.

- [ ] **Step 2E-3: Commit test changes (Commit 2 of 2)**

```bash
git add tests/unit/test_peft_method_protocol.py tests/unit/test_eval_runner.py tests/unit/test_cli.py tests/integration/test_peft_qlora_real.py
git commit -m "test(peft): flip qlora-reject tests; add load_from_disk + channel-adapter assertions; extend GPU roundtrip (#98)"
```

**Acceptance for this commit:**
- `test_qlora_adapter_supports_checkpoint_load_from_disk_true` exists and passes.
- `test_peft_method_protocol_declares_load_from_disk` exists and passes.
- `test_lora_adapter_load_from_disk_delegates_to_load_lora` exists and passes.
- `test_qlora_adapter_load_from_disk_delegates_to_load_qlora` exists and passes.
- `test_run_eval_dispatches_qlora_from_disk` exists and passes.
- `test_run_eval_lora_calls_load_channel_adapter` exists and passes.
- `test_eval_command_accepts_qlora_method` exists and passes.
- `grep "eval.runner.load_lora" tests/unit/test_eval_runner.py` — zero matches.
- `grep "peft_adapters.lora.load_lora" tests/unit/test_eval_runner.py` — six matches.

---

## Task 3: Manual gate + open PR

**Files:** none (commands + PR draft)

This task is the full acceptance gate from spec §5 followed by opening the PR. Orchestrator runs commands directly; no subagent dispatch.

- [ ] **Step 3A: Run the full non-GPU test suite (coverage gate)**

```bash
uv run pytest -m "not gpu"
```

Expected: all green AND coverage ≥ 80%. **Do NOT pass `--no-cov` here** — the 80% gate lives in `pyproject.toml` `addopts` (`--cov-fail-under=80`), so `--no-cov` would silently skip the gate. Intermediate subset runs in Tasks 1–2 use `--no-cov` (a subset cannot reach 80%); this final run must keep coverage ON. This must be the **full** suite (not just `tests/unit`) — GPU tests self-skip via their `requires_compatible_gpu`/`requires_checkpoint`/bnb skipif guards, mirroring CI's plain `uv run pytest`.

- [ ] **Step 3B: Run ruff lint/format checks AND mypy across all source and tests**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/custom_sam_peft
```

Expected: all three clean. **mypy is a CI gate** (ci.yml:44 runs `uv run mypy src/custom_sam_peft`); the new `load_from_disk` Protocol method and its `Any`-typed implementations must type-check. If mypy flags the `Any` returns or the Protocol/implementation variance, resolve before the PR — do not merge red.

- [ ] **Step 3C: Mechanical acceptance-criteria verification**

Run each of the following and confirm the expected output:

```bash
# §5 item 1: QloraAdapter.supports_checkpoint_load_from_disk() returns True
uv run python -c "from custom_sam_peft.peft_adapters import QloraAdapter; assert QloraAdapter().supports_checkpoint_load_from_disk() is True; print('OK: QloraAdapter supports_checkpoint_load_from_disk is True')"

# §5 item 2: both adapters satisfy isinstance(adapter, PEFTMethod)
uv run python -c "
from custom_sam_peft.peft_adapters import LoraAdapter, QloraAdapter, PEFTMethod
assert isinstance(LoraAdapter(), PEFTMethod)
assert isinstance(QloraAdapter(), PEFTMethod)
assert hasattr(LoraAdapter(), 'load_from_disk')
assert hasattr(QloraAdapter(), 'load_from_disk')
print('OK: both adapters pass isinstance(PEFTMethod) and have load_from_disk')
"

# §5 item 11: no bitsandbytes import at module level
grep "bitsandbytes" src/custom_sam_peft/peft_adapters/__init__.py && echo "FAIL: bitsandbytes found" || echo "OK: no bitsandbytes at module level"

# §5 item 12: no module-level load_lora / load_qlora import
grep -n "^from custom_sam_peft.peft_adapters.lora import\|^from custom_sam_peft.peft_adapters.qlora import" src/custom_sam_peft/peft_adapters/__init__.py && echo "FAIL: module-level import found" || echo "OK: no module-level load_lora/load_qlora"

# §5 item 14: runner.py has no .method == literals
grep "\.method ==" src/custom_sam_peft/eval/runner.py && echo "FAIL: .method == found" || echo "OK: no .method == in runner.py"
```

Expected: all `OK` lines; no `FAIL` lines.

- [ ] **Step 3D: Push the branch**

```bash
git push -u origin HEAD   # current worktree branch: worktree-qlora-eval-disk-load-98
```

- [ ] **Step 3E: Confirm available labels**

```bash
gh label list
```

Confirm `hardening-followup` exists. If not, create it:

```bash
gh label create hardening-followup --description "Follow-up tasks from the hardening audit" --color "e4e669"
```

- [ ] **Step 3F: Open the PR**

```bash
gh pr create \
  --assignee @me \
  --label hardening-followup \
  --title "feat(peft): QLoRA checkpoint disk-load in eval/runner.py via PEFTMethod.load_from_disk (#98)" \
  --body "$(cat <<'EOF'
**Spec:** [`docs/superpowers/specs/2026-05-24-qlora-eval-disk-load-design.md`](docs/superpowers/specs/2026-05-24-qlora-eval-disk-load-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-24-qlora-eval-disk-load-plan.md`](docs/superpowers/plans/2026-05-24-qlora-eval-disk-load-plan.md)

## Summary

- Adds `load_from_disk(wrapper, dirpath) -> wrapper` to the `PEFTMethod` Protocol; `LoraAdapter` delegates lazily to `load_lora`, `QloraAdapter` delegates lazily to `load_qlora` — bitsandbytes isolation preserved.
- Flips `QloraAdapter.supports_checkpoint_load_from_disk()` from `False` → `True`; removes the `ValueError` guard in `run_eval` that blocked QLoRA disk-load.
- Wires `_load_channel_adapter` into the `run_eval` disk-load path (latent bug fix: N-channel checkpoints in eval were silently not restoring the channel adapter).

## Commits

1. `feat(peft): add load_from_disk protocol method; wire QLoRA eval disk-load + channel adapter (#98)`
2. `test(peft): flip qlora-reject tests; add load_from_disk + channel-adapter assertions; extend GPU roundtrip (#98)`

## Test plan

- `uv run pytest -m "not gpu"` — green, coverage ≥ 80%
- `uv run ruff check src/ tests/` — clean
- `uv run ruff format --check src/ tests/` — clean
- `uv run mypy src/custom_sam_peft` — clean
- `tests/integration/test_peft_qlora_real.py::test_save_load_qlora_roundtrip` extended with forward-output parity assertion — GPU verification deferred to `gpu_local` tier

## Closes

Closes #98 — QLoRA checkpoint disk-load in eval/runner.py via PEFTMethod.load_from_disk.
EOF
)"
```

- [ ] **Step 3G: Surface the PR URL**

Capture and report the PR URL printed by `gh pr create`. Orchestrator then enters its idle phase.

**Acceptance:**
- All Step 3A–3C gates pass (green suite, clean ruff, all `OK` checks).
- PR is open against `main` from `worktree-qlora-eval-disk-load-98` with `Closes #98` in the description.
- PR has `@me` as assignee and the `hardening-followup` label.

---

## Spec §5 acceptance criteria — traceability

| Spec §5 item | Plan step that covers it |
| --- | --- |
| `QloraAdapter().supports_checkpoint_load_from_disk()` returns `True` | Step 1A-3 (source), Step 2A-2 (test), Step 3C (mechanical check) |
| `LoraAdapter().load_from_disk` and `QloraAdapter().load_from_disk` exist and pass `isinstance` | Steps 1A-4, 1A-5 (source), Step 2A-3, 2A-4 (delegation tests), Step 3C |
| `LoraAdapter().load_from_disk` calls `load_lora` | Step 2A-3 (mock delegation test) |
| `QloraAdapter().load_from_disk` calls `load_qlora` | Step 2A-4 (mock delegation test) |
| `run_eval` with `peft_method='qlora'`, `model=None` no longer raises `ValueError` | Step 1B-2 (guard deleted), Step 2B-1 (flipped test) |
| `run_eval` with `peft_method='qlora'`, `model=None` calls `load_qlora` | Step 2B-1 (assertion in `test_run_eval_dispatches_qlora_from_disk`) |
| `run_eval` with `peft_method='qlora'`, `model=None` calls `_load_channel_adapter` | Step 2B-1 (assertion in `test_run_eval_dispatches_qlora_from_disk`) |
| `run_eval` with `peft_method='lora'`, `model=None` calls `_load_channel_adapter` | Step 2B-2 (`test_run_eval_lora_calls_load_channel_adapter`) |
| `run_eval` with `model=<pre-loaded>` does NOT call `_load_channel_adapter` | Covered by existing `test_run_eval_accepts_prebuilt_val_dataset_and_model` (unchanged; `if model is None:` guard remains) |
| CLI does NOT produce "only LoRA" message for QLoRA config | Step 2C-1 (`test_eval_command_accepts_qlora_method`) |
| No `bitsandbytes` at module level in `peft_adapters/__init__.py` | Step 1A-7, Step 3C |
| No `load_lora`/`load_qlora` at module level in `peft_adapters/__init__.py` | Step 1A-7, Step 3C |
| `test_qlora_adapter_supports_checkpoint_load_from_disk_true` passes | Step 2A-2 |
| `test_eval_runner_does_not_branch_on_method_name` still passes | Step 1B-6 (guard check), Step 2A-5 (full protocol test suite) |
| `uv run pytest -m "not gpu"` — fully green | Step 3A |
| `uv run ruff check src/ tests/` — clean | Step 3B |
| `uv run ruff format --check src/ tests/` — clean | Step 3B |
| (GPU only) `test_save_load_qlora_roundtrip` forward-output parity passes | Step 2D-2 (code written); **GPU verification deferred to `gpu_local` tier — do not claim passed** |

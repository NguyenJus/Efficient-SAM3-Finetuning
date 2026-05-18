# Implementation Plan: Fix roi_align dtype mismatch under bfloat16

**Date:** 2026-05-17  
**Spec:** `docs/superpowers/specs/fix-roi-align-dtype-mismatch.md`  
**Difficulty:** L (low — ~30 lines total, no logic branches, clear precedent)

---

## Colab T4 status

8/9 tests passing as of the last run. This patch targets the 1 remaining failure:
`tests/integration/test_load_sam31_real.py::test_load_sam31_forward_to_canonical`
with `dtype="bfloat16"`.

Final validation: user re-runs `notebooks/colab_gpu_tests.ipynb` on Colab T4 and confirms 9/9.

---

## Rationale

sam3's `geometry_encoders.py::_encode_boxes` calls `torchvision.ops.roi_align` with rois cast to fp32 via `.float()`, but `img_feats` is bf16 when the model is cast to bf16. torchvision's C++ kernel requires both tensors to share dtype. We cannot modify sam3 (installed package), and we cannot use `torch.autocast` (it collides with sam3's internal `autocast(enabled=False)` in `decoder.py::forward_ffn`, as fixed in PR #13). The solution is a module-level monkey-patch on `torchvision.ops.roi_align` that casts `boxes` to `input.dtype` before delegating to the original.

---

## Tasks

### Task 1 — Add `_patch_roi_align_dtype()` to `src/esam3/models/sam3.py`  
**Difficulty:** L  
**File:** `src/esam3/models/sam3.py`  
**Diff size:** ~25 lines inserted  
**Constraints satisfied:** 1, 2, 3, 4

Insert the function immediately after the existing `_patch_pos_enc_dtype` (`src/esam3/models/sam3.py:255-310`) so the two patch helpers are adjacent. Follow the idempotent sentinel pattern that this codebase uses for monkey-patches — gate on `getattr(tvo, "_esam3_roi_align_dtype_patched", False)`, set it to `True` after patching, store the original as a closure variable. The docstring must state: (a) why we cannot touch sam3 source, (b) why autocast is prohibited (PR #13 / `decoder.py::forward_ffn`), (c) that both list and tensor `boxes` forms are handled.

Wrapper logic (exact):

```python
def _patch_roi_align_dtype() -> None:
    import torchvision.ops as tvo
    if getattr(tvo, "_esam3_roi_align_dtype_patched", False):
        return
    _original = tvo.roi_align

    def _roi_align_dtype_aware(input, boxes, *args, **kwargs):
        if isinstance(boxes, (list, tuple)):
            boxes = type(boxes)(
                b.to(dtype=input.dtype) if b.dtype != input.dtype else b
                for b in boxes
            )
        elif hasattr(boxes, "dtype") and boxes.dtype != input.dtype:
            boxes = boxes.to(dtype=input.dtype)
        return _original(input, boxes, *args, **kwargs)

    tvo.roi_align = _roi_align_dtype_aware
    tvo._esam3_roi_align_dtype_patched = True
```

**Validation:** `python -c "from esam3.models.sam3 import _patch_roi_align_dtype; _patch_roi_align_dtype(); import torchvision.ops; assert getattr(torchvision.ops, '_esam3_roi_align_dtype_patched', False)"`

---

### Task 2 — Call `_patch_roi_align_dtype()` from `load_sam31`  
**Difficulty:** L  
**File:** `src/esam3/models/sam3.py`  
**Diff size:** 1 line inserted  
**Constraints satisfied:** 3, 4

In `load_sam31`, after the `raw_model.to(dtype=...)` block and before `adapter = _Sam3ImageAdapter(...)`, insert:

```python
    _patch_roi_align_dtype()
```

This ensures the patch is in place before any forward pass through sam3's geometry encoder.

**Validation:** `python -m pytest tests/unit/test_sam3_roi_align_patch.py -x -q` (added in Task 3)

---

### Task 3 — Add `tests/unit/test_sam3_roi_align_patch.py`  
**Difficulty:** L  
**File:** `tests/unit/test_sam3_roi_align_patch.py` (new file)  
**Diff size:** ~70 lines  
**Constraints satisfied:** 4

Four test functions, all CPU-only (no `pytest.mark.gpu`):

1. `test_list_rois_dtype_mismatch_real_kernel` — fp32 input (1,1,4,4), fp16 rois as list of one (1,4) tensor, output_size=2. Assert call succeeds and `output.dtype == torch.float32`.
2. `test_tensor_rois_dtype_mismatch_mock` — fp32 input, fp16 rois as (1,5) Tensor. Use `unittest.mock.patch("torchvision.ops.roi_align._original"` — actually, patch the closure by temporarily replacing `_original` via `unittest.mock.MagicMock`. Simpler: unpatch, repatch inside the test using a mock as the original, then restore. Assert the mock received boxes with `dtype == torch.float32`.
3. `test_same_dtype_passthrough` — fp32 input + fp32 rois (list form). Assert output matches a direct unpatched `torchvision.ops.roi_align` call on the same inputs (run before calling `_patch_roi_align_dtype`).
4. `test_idempotency` — call `_patch_roi_align_dtype()` twice; capture `torchvision.ops.roi_align` after first call; assert it is `is` the same object after second call.

Important: `test_same_dtype_passthrough` must run the unpatched comparison first, then apply the patch. Either use test ordering or reset the sentinel in a fixture. The cleanest approach is to save `torchvision.ops.roi_align` before patching, call both variants, then assert equality of outputs. Do not rely on test execution order.

**Validation:** `python -m pytest tests/unit/test_sam3_roi_align_patch.py -v`

---

## Execution order

Run tasks sequentially: Task 1 → Task 2 → Task 3.

Tasks 1 and 2 are in the same file; do them together. Task 3 is independent once the function exists.

---

## Final check

```bash
python -m pytest tests/unit/test_sam3_roi_align_patch.py -v
python -m pytest tests/unit/ -x -q --ignore=tests/unit/test_sam3_roi_align_patch.py
```

Confirm no regressions. The integration test (`test_load_sam31_forward_to_canonical` with bf16) is validated only on Colab T4 — that is the user's final acceptance gate (9/9).

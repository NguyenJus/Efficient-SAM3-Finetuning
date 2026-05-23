# spec/issue-79-older-cuda-feasibility — Older CUDA/Driver/CC Feasibility Write-Up

**Status:** Draft (2026-05-23)
**Tracking:** [#79](https://github.com/NguyenJus/custom-sam-peft/issues/79)
**Scope:** Produce a desk-research feasibility write-up answering issue #79's questions about
supporting older CUDA toolkits, NVIDIA drivers, and GPU compute capabilities than the current
floor. The deliverable is a committed markdown research doc (`docs/research/`) plus a comment
posted on issue #79. No code, config, test, or CI file in the repository changes in this PR.

---

## 1. Goals & Non-Goals

### Goals

- Post a well-sourced comment on issue #79 whose body is the feasibility write-up.
- Commit the same write-up to `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md` so it
  is grep-able and durable across issue-tracker migrations.
- Enumerate potential follow-up work (tighter skip messages, cu118 CI matrix, LoRA-only path docs)
  as issue candidates inside the write-up — to be filed by the user after reviewing, NOT in this PR.
- Open a PR that adds only that one research file and links #79.

### Non-Goals

- Filing follow-up GitHub issues — only enumerating them in the write-up.
- Any change to `src/custom_sam_peft/`, `tests/`, `configs/`, `.github/workflows/`, or any existing
  doc other than the new research file.
- Rewording `conftest.py` CC-7.5 skip messages or any other code-adjacent edits.
- Expanding the CI matrix (e.g. adding a cu118 job).
- Running practical install verification beyond `uv pip install --dry-run`. No actual installs into
  the project environment.
- GPU execution of any kind.

---

## 2. Artifacts

| # | Path | Disposition | Notes |
|---|------|-------------|-------|
| 1 | `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md` | **NEW** | Only file changed in the PR. |
| 2 | Pull request adding artifact 1 | opened by implementer | PR title: `docs(research): older CUDA/driver/CC feasibility (#79)`. PR body links #79. |
| 3 | Comment on issue #79 | posted after PR merges | Body = contents of artifact 1 plus a "See merged write-up at \<permalink\>" footer. |

The `docs/research/` directory is new; the implementer creates it. No other directory is created.

---

## 3. Research Method

The implementer performs desk research and a dependency probe. No GPU execution, no real installs
into the project environment.

**Torch wheel availability.**

Run `uv pip install --dry-run` against both the cu118 and cu124 wheel indexes to confirm which
`torch>=2.4` wheels are available for Python 3.12 / Linux x86_64:

```
uv pip install --dry-run --index-url https://download.pytorch.org/whl/cu118 'torch>=2.4'
uv pip install --dry-run --index-url https://download.pytorch.org/whl/cu124 'torch>=2.4'
```

Record the resolved version and note which CUDA arches (`sm_XX`) each wheel ships for
(visible from the wheel filename suffix, e.g. `+cu118`). These commands are read-only probes;
`--dry-run` ensures nothing is installed.

**Bitsandbytes compatibility.**

Confirm bnb ≥ 0.43 compatibility with CUDA 11.8 vs. 12.x by reading the bitsandbytes README /
release notes as primary source — not folklore. Identify the 4-bit kernel compute-capability floor
from that source. Note that bnb is an optional extra in `pyproject.toml` (`qlora = ["bitsandbytes>=0.43"]`,
line 31) — the CC floor only matters when the QLoRA path is selected.

**sam3 transitive kernels.**

Inspect the pinned sam3 commit
(`sam3 @ git+https://github.com/facebookresearch/sam3@2814fa619404a722d03e9a012e083e4f293a4e53`,
`pyproject.toml:26`) for any custom CUDA kernels or ops beyond what stock torch provides (e.g.
custom `.cu` files, `torch.ops.load_library`, `_C` extension modules). If sam3 ships no custom
kernels, state that explicitly; its CC floor is then dictated entirely by the torch wheel.

**NVIDIA driver floor.**

Cite NVIDIA's published CUDA-to-minimum-driver table for CUDA 11.8 and CUDA 12.x (the
CUDA Toolkit Release Notes matrix). Record the minimum driver version required to run each toolkit.

**bf16 autocast.**

Native bf16 requires CC ≥ 8.0 (Ampere). `src/custom_sam_peft/train/loop.py:65–71` defines
`_autocast_ctx`, which selects `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` when
`cfg.model.dtype == "bfloat16"` (the default per `config/schema.py:48`). Note the implication:

- T4 (CC 7.5): bf16 is emulated; training runs but is slower / may produce different numerics.
- V100 (CC 7.0): does not support bf16 natively; torch.autocast on bf16 falls back to fp32 for
  ops without bf16 kernels; results may be silently degraded.

State this in the write-up under §3 (Compute capability floor).

**Path split.**

Produce all results independently for the two installation paths:

- **LoRA-only path** — bnb extra NOT installed (`pip install custom-sam-peft` without `[qlora]`).
- **QLoRA path** — bnb extra installed (`pip install custom-sam-peft[qlora]`).

---

## 4. Write-Up Structure

The research file at `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md` uses this
section order. The implementer fills in all findings; the spec describes the shape only.

### TL;DR

3–5 lines. One-sentence recommendation (support / partial / don't support). Two bullets — one per
path — summarizing the floor for each.

### §1 — CUDA toolkit floor

Answers issue #79 Q1. Cites the `uv pip install --dry-run` results for both indexes. States the
minimum CUDA toolkit version at which a `torch>=2.4` wheel is available for each path.

### §2 — Driver floor

Answers issue #79 Q2. Cites the NVIDIA CUDA→driver matrix. States the minimum driver version
required to run the toolkit floor identified in §1, for each path.

### §3 — Compute capability floor

Answers issue #79 Q3. Split by path:

- **LoRA-only**: floor set by torch wheel arch list; state the lowest `sm_XX` in the wheel.
- **QLoRA**: floor set by max(torch arch floor, bnb 4-bit kernel floor). Cite bnb release notes for
  the exact CC floor; explain what happens below that floor (module unavailable vs. runtime error).

Explain the bf16 autocast implication for T4 (CC 7.5) and V100 (CC 7.0) as documented in §3 of
the research method above.

### Compatibility matrix

One markdown table. Rows = {LoRA-only path, QLoRA path}. Columns:

| Path | Python | torch | min CUDA toolkit | min driver | min CC | status |
|------|--------|-------|-----------------|-----------|--------|--------|

Status values: `feasible`, `partial`, `not feasible`.

A `partial` status is accompanied by a footnote explaining what partial support means (e.g.,
"LoRA runs; bf16 is emulated below CC 8.0").

### Recommendation

Exactly one of: **support** / **partial** / **don't support**, per the rubric in §5. State which
bucket and why in 2–4 sentences.

### Follow-up issue candidates

Only present when recommendation ≠ "don't support". One bullet per candidate, format:

> `**<Title>** — <one-line scope>`

Likely candidates (the implementer confirms or refines based on findings):

- Tighten the `tests/conftest.py` CC-7.5 skip message to distinguish bnb-bound vs. SAM-bound
  requirements.
- Add a cu118 wheel index to the Docker install / CI matrix documentation.
- Document the LoRA-only-on-older-cards path in `docs/README-dev.md`.

These are candidates for the user to file later — they are NOT filed in this PR.

### Sources

Short appendix of links:
- PyTorch wheel download index for cu118 and cu124.
- bitsandbytes release notes or README (the specific section/tag that documents the 4-bit CC floor).
- NVIDIA CUDA Toolkit Release Notes (the driver compatibility table).
- sam3 pinned commit on GitHub (the `@2814fa619404a722d03e9a012e083e4f293a4e53` ref).

---

## 5. Recommendation Rubric

The write-up MUST state explicitly which bucket applies and cite the rubric.

| Bucket | Condition |
|--------|-----------|
| **Support** | Both LoRA and QLoRA paths can be lowered without adding > 1 wheel index, > 1 CI matrix row, or any code branching. |
| **Partial** | One path (likely LoRA-only) can be lowered cheaply; the other (QLoRA) hits a hard kernel CC floor. |
| **Don't support** | Even the cheapest lowering requires CI matrix expansion + code branching + new test surface. |

If the rubric produces "partial," the write-up specifies which path is feasible and which is not,
and why.

---

## 6. Workflow / Handoff

**Implementer (this PR).**

1. Create `docs/research/` directory; write
   `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md` with the full write-up (§4
   structure, all findings filled in).
2. Run the lint gate. Repo pre-commit (`.pre-commit-config.yaml`) only configures `ruff` (Python) and
   `nbstripout` (notebooks); both are no-ops for a markdown-only change. Confirm with
   `uv run pre-commit run --files docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`.
3. Commit the single new file; push to `research/79-older-cuda-feasibility`.
4. Open a PR with title `docs(research): older CUDA/driver/CC feasibility (#79)` and body that
   links `#79`. Pass `--assignee @me --label documentation` (or the closest existing label).

**Reviewer (next session).**

Verify sources, table↔prose consistency, and rubric (§7 below). No code to test, no CI to run.

**Post-merge (orchestrator).**

After the PR merges to `main`:

1. Resolve the merge commit SHA for the PR so the comment uses a commit-pinned permalink (not a
   branch-tip URL):

   ```
   SHA=$(gh pr view <pr-num> --json mergeCommit -q .mergeCommit.oid)
   ```

2. Build the comment body in a temp file: the contents of the merged file followed by a footer:

   ```
   ---
   Merged write-up: https://github.com/NguyenJus/custom-sam-peft/blob/$SHA/docs/research/2026-05-23-issue-79-older-cuda-feasibility.md
   ```

3. Post once: `gh issue comment 79 --body-file <tmp>`. Verify the comment renders correctly.

4. Do NOT close issue #79. The user reviews the comment and any enumerated follow-ups separately;
   closing is an explicit out-of-scope decision (§8).

---

## 7. Verification

The reviewer (next session) checks the following. No code to run; no CI.

- [ ] Every cited fact in §1, §2, §3, and the Sources appendix has a working URL or a precise
  document reference (section + version number).
- [ ] The compatibility matrix values are consistent with the prose in §1–§3. No cell contradicts
  a statement in the body.
- [ ] The Recommendation section cites the rubric from §5 and states which bucket applies with a
  reason.
- [ ] Follow-up issue candidates (if present) each name exactly one file or one scope of work.
- [ ] The `uv pip install --dry-run` commands are reproduced verbatim (or with their actual output)
  somewhere in the write-up so the finding is reproducible.
- [ ] The bf16 / CC 7.5 / CC 7.0 implication is addressed in §3 of the write-up.

---

## 8. Out of Scope

- Filing follow-up GitHub issues — only enumerating them in the write-up.
- Any change to `src/custom_sam_peft/`, `tests/`, `configs/`, `.github/workflows/`, or any doc
  other than the new research file.
- Rewording `tests/conftest.py` CC-7.5 skip messages (enumerated as a candidate only).
- Adding a cu118 CI matrix job (enumerated as a candidate only).
- Editing `docs/README-dev.md` (enumerated as a candidate only).
- Running `uv pip install` without `--dry-run` or any install into the project environment.
- GPU execution of any kind.
- Closing issue #79. The user reviews the posted comment and follow-up candidates separately.
- Determining whether the recommendation should be acted on — that decision is the user's after
  reviewing the write-up and considering the follow-up issue candidates.

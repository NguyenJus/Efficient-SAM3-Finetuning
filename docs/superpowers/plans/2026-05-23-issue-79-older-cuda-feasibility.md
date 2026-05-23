# Older CUDA/Driver/CC Feasibility Write-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a desk-research markdown file at `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md` answering issue #79's questions about supporting older CUDA toolkits, drivers, and GPU compute capabilities, then open a PR and post the write-up as a comment on #79 after merge.

**Architecture:** This is a research-and-documentation task; no source code, config, test, or CI files are created or modified at any point. The implementer runs read-only probe commands (`uv pip install --dry-run`), performs desk research, drafts the write-up per the spec's §4 structure, and commits the single new markdown file. The orchestrator handles post-merge comment posting.

**Tech Stack:** Markdown, `uv pip install --dry-run`, `gh` CLI, NVIDIA/PyTorch/bitsandbytes documentation sources.

---

## File Map

| File | Disposition | Responsible party |
|------|-------------|-------------------|
| `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md` | **NEW** (implementer creates) | Implementer |
| PR adding the above | Opened by implementer | Implementer |
| Comment on issue #79 | Posted after PR merges | Orchestrator |

No other files are created or modified.

---

## Phase 1 — Research (Probe + Desk Research)

### Task 1: Run torch wheel availability probes

**Files:**
- No files modified; output recorded in scratch (implementer should keep terminal output to paste into write-up later)

- [ ] **Step 1: Run cu118 dry-run probe**

  ```bash
  uv pip install --dry-run \
    --index-url https://download.pytorch.org/whl/cu118 \
    'torch>=2.4'
  ```

  Expected: uv resolves and prints the torch wheel it would install (e.g. `torch-2.X.X+cu118-cp312-cp312-linux_x86_64.whl`). Nothing is installed. Capture the full output — the wheel filename suffix (`+cu118`) and the resolved version number are needed for the write-up.

- [ ] **Step 2: Run cu124 dry-run probe**

  ```bash
  uv pip install --dry-run \
    --index-url https://download.pytorch.org/whl/cu124 \
    'torch>=2.4'
  ```

  Expected: Similar output with `+cu124` suffix. Capture the resolved version and wheel filename.

- [ ] **Step 3: Record probe outputs**

  Note down (in a scratch buffer or comment block):
  - cu118 resolved torch version + wheel filename
  - cu124 resolved torch version + wheel filename
  - Whether both probes succeeded or either failed with "no matching distribution"

  These outputs will be embedded verbatim in §1 of the write-up.

**Checkpoint A — Research phase gate (torch probes):** Both probe commands have been run. The resolved torch wheel version and `+cuXXX` suffix are captured for each. If either probe fails (no wheel found), that itself is the finding — record the error output.

---

### Task 2: Desk research — bitsandbytes CC floor

**Files:**
- No files modified; findings recorded in scratch

- [ ] **Step 1: Read bitsandbytes release notes / README for 4-bit kernel CC floor**

  Primary sources to check (in order of preference):
  1. bitsandbytes GitHub README: https://github.com/TimDettmers/bitsandbytes — search for "compute capability" or "sm_" or "CC" under the requirements section.
  2. bitsandbytes releases page: https://github.com/TimDettmers/bitsandbytes/releases — check ≥ 0.43 release notes for any CC floor statement.
  3. `pyproject.toml` in bitsandbytes repo for `torch` lower-bound / wheel arch list.

  Confirm: What is the **minimum compute capability** for 4-bit quantization (NF4/FP4) in bnb ≥ 0.43? (Expected to be CC 7.5 / sm_75 based on Turing, but cite the actual source — do not assume.)

- [ ] **Step 2: Confirm bnb CUDA 11.8 compatibility**

  In the same sources, confirm that bnb ≥ 0.43 supports CUDA 11.8 wheel indexes (or requires 12.x). Note any explicit statement about minimum CUDA toolkit version for bnb.

- [ ] **Step 3: Record findings + URL**

  Write down:
  - The bnb ≥ 0.43 4-bit CC floor (e.g. `sm_75` / CC 7.5)
  - Whether bnb ≥ 0.43 supports CUDA 11.8
  - The exact URL + section/anchor (or release tag + section heading) used as the source

**Checkpoint B — Research phase gate (bnb):** CC floor for bnb 4-bit is recorded with a citable URL or precise document reference (tag + heading). CUDA 11.8 compatibility confirmed or refuted from the same source.

---

### Task 3: Desk research — sam3 custom kernel audit

**Files:**
- No files modified; findings recorded in scratch

- [ ] **Step 1: Open the pinned sam3 commit on GitHub**

  URL: `https://github.com/facebookresearch/sam3/tree/2814fa619404a722d03e9a012e083e4f293a4e53`

  Browse the top-level directory listing. Look for:
  - Any `.cu` or `.cuh` files anywhere in the tree
  - Any `setup.py` / `setup.cfg` referencing `ext_modules` or `CUDAExtension`
  - Any `torch.ops.load_library(...)` call in Python source
  - Any `_C` extension module imported in `__init__.py` or similar

- [ ] **Step 2: Confirm finding**

  One of two outcomes:
  - **No custom kernels**: state "sam3 at commit `2814fa6` ships no custom CUDA kernels or ops; its CC floor is dictated entirely by the torch wheel."
  - **Custom kernels found**: list the files, the `sm_XX` targets they compile for, and the resulting CC floor.

  Record this statement verbatim — it will be quoted in §3 of the write-up.

**Checkpoint C — Research phase gate (sam3):** The sam3 commit has been browsed. Custom kernel presence or absence is recorded with the commit URL as source.

---

### Task 4: Desk research — NVIDIA driver floor

**Files:**
- No files modified; findings recorded in scratch

- [ ] **Step 1: Look up the NVIDIA CUDA Toolkit Release Notes driver compatibility table**

  Primary source: NVIDIA CUDA Toolkit Release Notes — https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

  Find the table "Table 1. CUDA Toolkit and Compatible Driver Versions" (or equivalent heading). Record:
  - Minimum driver version for CUDA 11.8 (Linux)
  - Minimum driver version for CUDA 12.x (use the lowest 12.x row, typically 12.0 or 12.1) (Linux)

- [ ] **Step 2: Record findings + URL**

  Write down:
  - CUDA 11.8 → minimum driver version (e.g. `520.61.05`)
  - CUDA 12.x (floor) → minimum driver version
  - The exact URL and section/table heading used as the source

**Checkpoint D — Research phase gate (driver table):** Both driver floor values are recorded with the NVIDIA URL and table name cited.

---

## Phase 2 — Drafting

### Task 5: Create `docs/research/` directory and scaffold the write-up file

**Files:**
- Create: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Create the directory and empty file**

  ```bash
  mkdir -p /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility/docs/research
  touch /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility/docs/research/2026-05-23-issue-79-older-cuda-feasibility.md
  ```

- [ ] **Step 2: Write the file skeleton with section headings**

  Write the following skeleton to the file (the implementer will fill each section in the subsequent tasks):

  ```markdown
  # Older CUDA / Driver / Compute Capability Feasibility

  > Research write-up for [issue #79](https://github.com/NguyenJus/custom-sam-peft/issues/79)
  > Date: 2026-05-23

  ## TL;DR

  <!-- Filled in last — depends on §1–§3 and matrix findings -->

  ## §1 — CUDA Toolkit Floor

  ## §2 — Driver Floor

  ## §3 — Compute Capability Floor

  ## Compatibility Matrix

  ## Recommendation

  ## Follow-up Issue Candidates

  ## Sources
  ```

---

### Task 6: Draft §1 — CUDA toolkit floor

**Files:**
- Modify: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Fill in §1 using probe results from Task 1**

  §1 must contain:
  - The verbatim or copy-pasted `uv pip install --dry-run` commands and their output (resolved wheel name + version for both cu118 and cu124).
  - A clear statement of the minimum CUDA toolkit version at which a `torch>=2.4` wheel is available for Python 3.12 / Linux x86_64.
  - Separate treatment for the two paths: LoRA-only (only torch needed) and QLoRA (torch + bnb ≥ 0.43). For §1, the bnb CUDA floor from Task 2 informs whether the QLoRA path can use cu118 or requires cu124.

  Example structure (implementer fills in actual values):

  ```markdown
  ## §1 — CUDA Toolkit Floor

  ### Probe commands and output

  LoRA-only path and QLoRA path share the same torch wheel index. Both
  `uv pip install --dry-run` commands were run against Python 3.12 / Linux x86_64:

  **cu118 index:**
  ```
  $ uv pip install --dry-run --index-url https://download.pytorch.org/whl/cu118 'torch>=2.4'
  <actual output here>
  ```

  **cu124 index:**
  ```
  $ uv pip install --dry-run --index-url https://download.pytorch.org/whl/cu124 'torch>=2.4'
  <actual output here>
  ```

  ### Finding

  `torch>=2.4` wheels are available for CUDA <version> and above.
  Minimum CUDA toolkit for the **LoRA-only path**: CUDA <X.Y>.
  Minimum CUDA toolkit for the **QLoRA path**: CUDA <X.Y> (further constrained by bnb — see §3).
  ```

---

### Task 7: Draft §2 — Driver floor

**Files:**
- Modify: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Fill in §2 using NVIDIA driver table findings from Task 4**

  §2 must contain:
  - The minimum driver version for CUDA 11.8 (Linux).
  - The minimum driver version for the CUDA 12.x floor (Linux).
  - A citation of the NVIDIA CUDA Toolkit Release Notes table (URL + table name).
  - Separate rows or sentences for the LoRA-only path and QLoRA path (they may differ if bnb forces a higher CUDA toolkit).

  Example structure:

  ```markdown
  ## §2 — Driver Floor

  Source: NVIDIA CUDA Toolkit Release Notes, "Table 1. CUDA Toolkit and Compatible Driver Versions"
  (<URL>).

  | CUDA toolkit | Minimum driver (Linux) |
  |---|---|
  | 11.8 | <version> |
  | 12.x (floor) | <version> |

  **LoRA-only path**: minimum driver <version> (requires CUDA <toolkit>).
  **QLoRA path**: minimum driver <version> (requires CUDA <toolkit>).
  ```

---

### Task 8: Draft §3 — Compute capability floor

**Files:**
- Modify: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Fill in §3 — LoRA-only CC floor**

  State the lowest `sm_XX` arch in the torch wheel (from the wheel filename metadata or PyTorch documentation for the identified version). This is the CC floor for the LoRA-only path.

- [ ] **Step 2: Fill in §3 — QLoRA CC floor**

  State the CC floor for bnb ≥ 0.43 4-bit kernels (from Task 2 findings). The QLoRA path floor = max(torch arch floor, bnb 4-bit kernel floor).

- [ ] **Step 3: Fill in §3 — sam3 kernel finding**

  State the sam3 custom-kernel finding verbatim (from Task 3). If no custom kernels, state that explicitly.

- [ ] **Step 4: Fill in §3 — bf16 autocast implications**

  §3 must explicitly address:
  - T4 (CC 7.5): bf16 is emulated — training runs but is slower / may produce different numerics. This is because `_autocast_ctx` in `src/custom_sam_peft/train/loop.py:65–71` selects `torch.autocast(dtype=torch.bfloat16)` by default (`cfg.model.dtype == "bfloat16"`, default per `config/schema.py:48`), and CC 7.5 does not have native bf16 hardware.
  - V100 (CC 7.0): does not support bf16 natively; `torch.autocast` on bf16 falls back to fp32 for ops without bf16 kernels; results may be silently degraded.

  Example paragraph:

  ```markdown
  ### bf16 autocast implication

  The default training dtype is `bfloat16` (`config/schema.py:48`). `_autocast_ctx` in
  `src/custom_sam_peft/train/loop.py:65–71` selects `torch.autocast(device_type="cuda",
  dtype=torch.bfloat16)` unconditionally when this dtype is configured.

  - **T4 (CC 7.5)**: bf16 is _emulated_ in software. Training completes but throughput is
    reduced and numerics may differ vs. Ampere.
  - **V100 (CC 7.0)**: no native bf16 support. `torch.autocast` falls back to fp32 for
    ops lacking bf16 kernels; results may be silently degraded.

  Users on pre-Ampere hardware should set `model.dtype: float16` in their config to avoid
  this degradation.
  ```

---

### Task 9: Draft the Compatibility Matrix

**Files:**
- Modify: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Fill in the compatibility matrix table**

  The table schema (from spec §4) is:

  ```markdown
  ## Compatibility Matrix

  | Path | Python | torch | min CUDA toolkit | min driver | min CC | status |
  |------|--------|-------|-----------------|-----------|--------|--------|
  | LoRA-only | ... | ... | ... | ... | ... | feasible / partial / not feasible |
  | QLoRA | ... | ... | ... | ... | ... | feasible / partial / not feasible |
  ```

  Fill in all cells using the findings from §1, §2, §3. Python version is 3.12 (confirmed from probe command). torch version is the resolved wheel version from Task 1.

- [ ] **Step 2: Add footnotes for any `partial` status cells**

  If either row has status `partial`, add a footnote directly below the table explaining what partial means. Example:

  ```markdown
  > *partial: LoRA training runs on CUDA 11.8 / CC 7.5 hardware, but bf16 is emulated
  > below CC 8.0 — set `model.dtype: float16` in config to avoid silent numeric degradation.*
  ```

---

### Task 10: Draft Recommendation section

**Files:**
- Modify: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Apply the rubric from spec §5 and write the Recommendation**

  The rubric (from spec §5):

  | Bucket | Condition |
  |--------|-----------|
  | **Support** | Both LoRA and QLoRA paths can be lowered without adding > 1 wheel index, > 1 CI matrix row, or any code branching. |
  | **Partial** | One path (likely LoRA-only) can be lowered cheaply; the other (QLoRA) hits a hard kernel CC floor. |
  | **Don't support** | Even the cheapest lowering requires CI matrix expansion + code branching + new test surface. |

  Determine which bucket applies based on the §1–§3 findings and matrix. Write 2–4 sentences stating which bucket and why, explicitly citing the rubric. Example:

  ```markdown
  ## Recommendation

  **Partial.**

  The LoRA-only path can be lowered to CUDA 11.8 / CC 7.5 by pointing users to the `+cu118`
  torch wheel index — no code branching or CI matrix expansion required. The QLoRA path
  cannot be lowered below CC <X.X> without hitting the bnb 4-bit kernel floor; lowering it
  further would require bnb to ship a CC <Y.Y> kernel, which is not the case as of bnb 0.43.
  Per the rubric (spec §5): one path lowers cheaply, the other hits a hard CC floor → **Partial**.
  ```

  (Implementer substitutes actual values and adjusts bucket as findings warrant.)

---

### Task 11: Draft Follow-up Issue Candidates and Sources

**Files:**
- Modify: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Write Follow-up Issue Candidates section**

  Only include this section if the recommendation is NOT "don't support". Each bullet uses the format from spec §4:

  ```markdown
  ## Follow-up Issue Candidates

  - **Tighten CC-7.5 skip message in `tests/conftest.py`** — Distinguish bnb-bound requirements
    (QLoRA path) from SAM-bound requirements so the skip reason is accurate for each path.
  - **Document cu118 wheel index in `docs/README-dev.md`** — Add an "older GPU" section
    explaining the LoRA-only path on CUDA 11.8 / CC 7.5 hardware.
  - **Add cu118 CI matrix documentation** — Describe what a cu118 CI job would cover and
    the cost/benefit tradeoff (candidate only; not filed in this PR).
  ```

  Confirm or refine these candidates based on actual findings. If a candidate is not warranted by the findings, omit it.

- [ ] **Step 2: Write Sources appendix**

  ```markdown
  ## Sources

  1. PyTorch wheel index — cu118: https://download.pytorch.org/whl/cu118
  2. PyTorch wheel index — cu124: https://download.pytorch.org/whl/cu124
  3. bitsandbytes release notes / README — 4-bit CC floor: <exact URL + section/tag used>
  4. NVIDIA CUDA Toolkit Release Notes — driver compatibility table: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
  5. sam3 pinned commit (2814fa6): https://github.com/facebookresearch/sam3/tree/2814fa619404a722d03e9a012e083e4f293a4e53
  ```

  Fill in the exact bnb URL and section anchor from Task 2 findings.

---

### Task 12: Draft TL;DR (last, depends on all other sections)

**Files:**
- Modify: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Fill in the TL;DR at the top of the file**

  TL;DR must be 3–5 lines. One-sentence recommendation. Two bullets — one per path — summarizing the floor. Example:

  ```markdown
  ## TL;DR

  **Partial support is feasible**: the LoRA-only path works on CUDA 11.8 / CC 7.5 hardware with
  no code changes; the QLoRA path is blocked by the bnb 4-bit kernel floor at CC <X.X>.

  - **LoRA-only path**: minimum CUDA <X.Y>, driver <version>, CC <X.X>. Status: `feasible`
    (bf16 emulated below CC 8.0 — use `float16` config instead).
  - **QLoRA path**: minimum CUDA <X.Y>, driver <version>, CC <X.X>. Status: `partial`
    (hard bnb 4-bit kernel floor; cannot lower further without bnb changes).
  ```

  (Implementer substitutes actual values and adjusts recommendation word to match Recommendation section.)

**Checkpoint E — Drafting phase gate:** All sections from spec §4 are present and non-empty in the file: TL;DR, §1, §2, §3, Compatibility Matrix, Recommendation, Follow-up Issue Candidates (if recommendation ≠ "don't support"), Sources. The matrix is fully populated. The Recommendation cites the rubric. The `uv pip install --dry-run` output is reproduced verbatim. The bf16/CC 7.5/CC 7.0 implication is addressed in §3.

---

## Phase 3 — Lint Gate

### Task 13: Run pre-commit lint check

**Files:**
- No files modified

- [ ] **Step 1: Run pre-commit against the new file**

  ```bash
  cd /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility && \
  uv run pre-commit run --files docs/research/2026-05-23-issue-79-older-cuda-feasibility.md
  ```

  Expected output: All hooks pass (ruff is a no-op for markdown; nbstripout is a no-op for markdown). If any hook modifies the file (e.g. a trailing-newline hook), stage the modification and note it.

- [ ] **Step 2: Verify the file is unchanged (or note any hook-triggered change)**

  ```bash
  git -C /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility \
    diff docs/research/2026-05-23-issue-79-older-cuda-feasibility.md
  ```

  Expected: no diff (file was not modified by pre-commit). If there is a diff, apply the fix and re-run pre-commit until clean.

---

## Phase 4 — Commit and PR

### Task 14: Commit the new file and open PR

**Files:**
- Commit: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

- [ ] **Step 1: Stage the new file**

  ```bash
  git -C /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility \
    add docs/research/2026-05-23-issue-79-older-cuda-feasibility.md
  ```

- [ ] **Step 2: Confirm only the one new file is staged**

  ```bash
  git -C /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility \
    status
  ```

  Expected: only `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md` is listed under "Changes to be committed". If any other file appears, do NOT commit — investigate and remove it from staging.

- [ ] **Step 3: Commit**

  ```bash
  git -C /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility \
    commit -m "$(cat <<'EOF'
  docs(research): older CUDA/driver/CC feasibility (#79)

  Adds desk-research write-up answering issue #79's questions about
  supporting older CUDA toolkits, NVIDIA drivers, and GPU compute
  capabilities. Covers LoRA-only and QLoRA paths; includes probe output,
  compatibility matrix, recommendation, and follow-up issue candidates.
  EOF
  )"
  ```

- [ ] **Step 4: Push to remote**

  ```bash
  git -C /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility \
    push -u origin research/79-older-cuda-feasibility
  ```

- [ ] **Step 5: Check that `documentation` label exists; create it if not**

  ```bash
  gh label list --repo NguyenJus/custom-sam-peft | grep documentation
  ```

  If not present:
  ```bash
  gh label create documentation \
    --description "Documentation changes" \
    --color 0075ca \
    --repo NguyenJus/custom-sam-peft
  ```

- [ ] **Step 6: Open the PR**

  ```bash
  gh pr create \
    --repo NguyenJus/custom-sam-peft \
    --title "docs(research): older CUDA/driver/CC feasibility (#79)" \
    --body "$(cat <<'EOF'
  ## Summary

  Adds `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md` — a desk-research
  write-up answering the questions in #79 about supporting older CUDA toolkits, NVIDIA
  drivers, and GPU compute capabilities than the current floor.

  **No source code, config, test, or CI files are changed.**

  Refs #79 — the write-up itself is the deliverable. The issue stays open after
  merge for user review of the findings and follow-up issue candidates. Do NOT
  use GitHub closing keywords (Closes/Fixes/Resolves) here.
  EOF
  )" \
    --assignee @me \
    --label documentation
  ```

  Record the PR number from the output — the orchestrator needs it for post-merge steps.

**Checkpoint F — PR phase gate:** PR is open with title `docs(research): older CUDA/driver/CC feasibility (#79)`. Only one file is changed in the diff. PR is assigned to `@me` and has the `documentation` label. No CI failures (pre-commit is the only check; it passed in Phase 3).

---

## Phase 5 — Reviewer Subagent

> **Orchestrator note:** Dispatch a separate reviewer subagent (min sonnet/high) after the PR is open. The reviewer does NOT merge.

### Task 15: Reviewer verifies spec §7 checklist

**Files:**
- Read-only: `docs/research/2026-05-23-issue-79-older-cuda-feasibility.md`

The reviewer checks every item in spec §7:

- [ ] Every cited fact in §1, §2, §3, and the Sources appendix has a working URL or a precise document reference (section + version number). The reviewer should open each URL and confirm it resolves.
- [ ] The compatibility matrix values are consistent with the prose in §1–§3. No cell contradicts a statement in the body.
- [ ] The Recommendation section cites the rubric from spec §5 and states which bucket applies with a reason.
- [ ] Follow-up issue candidates (if present) each name exactly one file or one scope of work.
- [ ] The `uv pip install --dry-run` commands are reproduced verbatim (or with their actual output) somewhere in the write-up so the finding is reproducible.
- [ ] The bf16 / CC 7.5 / CC 7.0 implication is addressed in §3 of the write-up.

**If any item fails:** The reviewer reports the specific failure to the orchestrator. The orchestrator dispatches the implementer subagent to fix the specific item, re-commit, and re-push. The reviewer re-checks only the fixed items.

**Checkpoint G — Reviewer phase gate:** All six spec §7 checklist items are confirmed by the reviewer subagent. No checklist item is left unverified.

---

## Phase 6 — Post-Merge (Orchestrator only)

> **Orchestrator note:** Do NOT delegate these steps to the implementer. The orchestrator performs them directly after the PR merges.

### Task 16: Post comment on issue #79

- [ ] **Step 1: Resolve the merge commit SHA**

  ```bash
  SHA=$(gh pr view <pr-num> --json mergeCommit -q .mergeCommit.oid)
  echo "Merge SHA: $SHA"
  ```

  Replace `<pr-num>` with the actual PR number recorded in Task 14 Step 6.

- [ ] **Step 2: Build the comment body in a temp file**

  Read the file from the worktree (it is identical to the merged content on `main` since the
  branch was up-to-date before merge; using the worktree path avoids a `git pull` race on the
  parent checkout):

  ```bash
  WRITE_UP_PATH=/home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility/docs/research/2026-05-23-issue-79-older-cuda-feasibility.md
  PERMALINK="https://github.com/NguyenJus/custom-sam-peft/blob/${SHA}/docs/research/2026-05-23-issue-79-older-cuda-feasibility.md"
  TMPFILE=$(mktemp /tmp/issue-79-comment-XXXXXX.md)
  { cat "$WRITE_UP_PATH"; printf '\n\n---\nMerged write-up: %s\n' "$PERMALINK"; } > "$TMPFILE"
  echo "Temp file: $TMPFILE"
  ```

  This step MUST run before Phase 7 close-out (which removes the worktree).

- [ ] **Step 3: Post the comment**

  ```bash
  gh issue comment 79 \
    --repo NguyenJus/custom-sam-peft \
    --body-file "$TMPFILE"
  ```

- [ ] **Step 4: Verify the comment renders**

  ```bash
  gh issue view 79 --repo NguyenJus/custom-sam-peft --comments | tail -40
  ```

  Confirm the last comment contains the write-up content and the permalink footer. The permalink URL must include the merge commit SHA (not `main` or a branch tip).

- [ ] **Step 5: Do NOT close issue #79**

  The user reviews the posted comment and follow-up candidates separately. Closing is explicitly out of scope (spec §8).

**Checkpoint H — Post-merge phase gate:** Comment is posted on issue #79. The comment includes the full write-up and a "Merged write-up: `<permalink>`" footer. The permalink URL contains the merge commit SHA. Issue #79 is still open.

---

## Phase 7 — Close-out (Orchestrator only — CLAUDE.md close-out protocol)

### Task 17: Close-out

- [ ] **Step 1: No-tag (no-ship: docs-only)**

  This PR is a docs-only change. Per CLAUDE.md close-out: "Skip no-ships (CI/dev-tooling/docs/lockfile-only/internal-refactor) — note in sign-off." **No git tag is created.**

- [ ] **Step 2: Kill any background processes**

  No dev servers or watchers were started for this task. Confirm:
  ```bash
  # Nothing to kill for this docs-only PR.
  echo "No background processes to kill."
  ```

- [ ] **Step 3: Fold branch log**

  ```bash
  ROOT=$(git -C /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility \
    rev-parse --show-toplevel | sed 's|/.worktrees/.*||')
  BRANCH_FILE=$(git -C /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility \
    rev-parse --abbrev-ref HEAD | tr '/' '-')
  printf '\n## %s (merged %s)\n' "$BRANCH_FILE" "$(date -I)" >> "$ROOT/logs/logs.md"
  cat "$ROOT/logs/$BRANCH_FILE.md" >> "$ROOT/logs/logs.md"
  rm "$ROOT/logs/$BRANCH_FILE.md"
  ```

- [ ] **Step 4: Remove the worktree**

  ```bash
  ROOT=$(git -C /home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility \
    rev-parse --show-toplevel | sed 's|/.worktrees/.*||')
  WORKTREE=/home/justin/projects/custom-sam-peft/.worktrees/research-79-older-cuda-feasibility
  cd "$ROOT" && git worktree remove "$WORKTREE"
  ```

- [ ] **Step 5: Sign off**

  Emit one-line sign-off: "Close-out complete: branch log folded, worktree removed. No tag created (no-ship: docs-only)."

---

## Verification Checkpoint Summary

| Checkpoint | Gate | Who verifies |
|-----------|------|--------------|
| A | Both `uv pip install --dry-run` commands run; torch wheel versions and suffixes captured. | Implementer |
| B | bnb ≥ 0.43 4-bit CC floor recorded with citable URL; CUDA 11.8 compat confirmed or refuted. | Implementer |
| C | sam3 custom-kernel audit complete; finding stated with commit URL as source. | Implementer |
| D | NVIDIA driver floor values for CUDA 11.8 and 12.x recorded with NVIDIA URL. | Implementer |
| E | All spec §4 sections present and populated; matrix filled; recommendation cites rubric; probe output verbatim; bf16 implications addressed. | Implementer |
| F | PR open; single-file diff; assigned; `documentation` label; no CI failures. | Implementer |
| G | All six spec §7 checklist items confirmed by reviewer subagent. | Reviewer subagent |
| H | Comment posted on #79 with write-up content + commit-pinned permalink; issue still open. | Orchestrator |

<!-- Append-only deferred-work log per ~/.claude/CLAUDE.md.
     Format: [TIMESTAMP] [ROLE] action | [DEFERRED] issue
     Never read during task execution. -->

[2026-05-16] [planner] [DEFERRED] revisit iscrowd handling after first real eval pass — v0 drops iscrowd=1 annotations entirely
[2026-05-16] [planner] [DEFERRED] named transform suites — let users pick "default" / "augmentation_heavy" / "geometric_only" from a menu instead of editing aug params
[2026-05-16] [implementer] [DEFERRED] task 10 — plan's determinism test assumed albumentations 1.x global-seed behavior; switched to compose.set_random_seed for 2.x. Plan doc should be updated.

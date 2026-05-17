[2026-05-17T18:31:34Z] [planner] training-loop spec deferred items | [DEFERRED] eval call site in spec/eval (architecture step 6) — Trainer must accept an injected Evaluator and call it at eval_every boundaries
[2026-05-17T18:31:34Z] [planner] training-loop spec deferred items | [DEFERRED] prompt_mode='bbox' training — currently rejected at Trainer.__init__; future spec/bbox-prompt-training
[2026-05-17T18:31:34Z] [planner] training-loop spec deferred items | [DEFERRED] early-stopping callback — MUST gate on current_box_hint_p < cfg.train.box_hint.early_stop_p_threshold
[2026-05-17T18:31:34Z] [planner] training-loop spec deferred items | [DEFERRED] multi-GPU / DDP / FSDP — Ray Train spec; single-device assumption is in _resolve_device() and wrapper.to(device)
[2026-05-17T18:31:34Z] [planner] training-loop spec deferred items | [DEFERRED] multiplex (multi-class-per-forward) — invariants: list[Tensor|None] is C=1 slice of list[list[Tensor|None]]; train_step's class loop is the only C=1 site
[2026-05-17T18:31:34Z] [planner] training-loop spec deferred items | [DEFERRED] cosine/exp box-hint schedule shapes — add BoxHintSchedule.shape literal when needed
[2026-05-17T18:31:34Z] [planner] training-loop spec deferred items | [DEFERRED] image-panel K parameter — hard-coded K=10 top-query merge for prediction visualization
[2026-05-17T18:31:34Z] [planner] training-loop spec deferred items | [DEFERRED] determinism flag — torch.use_deterministic_algorithms left False due to grad-checkpointing + bnb non-determinism; resume reproducibility comes from RNG-state restore

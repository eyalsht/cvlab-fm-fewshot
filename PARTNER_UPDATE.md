# Partner Update — Stage 1 complete, Stage 2 machinery ready

**Project:** Flow matching as a *layer* — few-shot classification over frozen encoder features
**Status:** Stage 1 finished and measured · Stage 2 machinery built and tested · the FM head itself is next

---

## 1. Where we landed, and why it matters

The full Stage 1 pipeline is built and the complete grid has run: 2 datasets × 2 encoders × 2 heads × 3 training-set
sizes × 3 seeds. The headline is not "the baselines work" — it is **the gap they exposed**, which is Stage 2's target.

**Top-1 accuracy, complete official test split, mean ± sample std over three seeds:**

| Dataset | Encoder | Head | K=5 | K=10 | Full |
|:--|:--|:--|--:|--:|--:|
| DTD (47) | ResNet-18 | Prototype | **48.3** ±1.3 | 51.7 ±0.2 | 58.8 |
| DTD (47) | ResNet-18 | Linear probe | 46.8 ±1.0 | **52.6** ±1.2 | **62.6** ±0.4 |
| Aircraft (100) | ResNet-18 | Prototype | 16.7 ±0.5 | 19.1 ±0.6 | 25.2 |
| Aircraft (100) | ResNet-18 | Linear probe | 20.7 ±0.3 | 26.6 ±0.7 | 36.8 ±0.2 |
| Aircraft (100) | DINOv2 ViT-S/14 | Prototype | 25.5 ±0.6 | 29.1 ±0.7 | 36.2 |
| Aircraft (100) | DINOv2 ViT-S/14 | Linear probe | **43.0** ±0.9 | **56.8** ±0.5 | **74.1** ±0.1 |

**Three things to take away:**

1. **The prototype gap.** Swapping ResNet-18 → DINOv2 on Aircraft roughly doubles the probe (36.8 → 74.1) and barely
   moves the prototype head (25.2 → 36.2): a 37.9-point spread on the *same features*. Better representations do not
   help a centroid rule at all.
2. **The diagnosis.** Those features are linearly separable but not *compact* around their class means — stretched
   along a roughly shared direction. Cosine-to-centroid reads that stretch as distance; a linear map does not. Written
   up in `docs/notes/prototype_gap_diagnosis.md`.
3. **A clean bias–variance crossover.** On DTD at K=5 the prototype *beats* the probe and loses by K=10 — with 5 images
   per class a mean is a better estimator than 47×512 fitted parameters. Good sign the protocol is sound.

Confusions are semantic, not noise: `C-47 → DC-3` at 48% (the same airframe under two names),
`BAE 146-300 → 146-200` at 44%, `dotted → polka-dotted` at 43%.

---

## 2. Decisions made (and why)

**Protocol — the biggest change.** The supervisor's Stage 1 spec replaced episodic N-way K-shot with **balanced
K-per-class subsets of the official train split**, scored on the *complete* official test split; the episode sampler
was deleted. 5-shot and 10-shot draws at one seed are seeded independently (`SeedSequence([seed, k])`), not nested —
nesting would correlate the two settings' errors and understate the spread. Train/val are **never merged**, which is
why Aircraft's `trainval` split is unreachable by construction.

**Prototype branch — Option A.** Image-derived prototypes, not zero-shot CLIP text prototypes; CLIP and the zero-shot
head were removed, and the baselines narrowed to the probe plus this one branch (the ridge head was dropped). His
formula, followed literally: normalize each feature → class mean → renormalize → cosine. The inner normalization is
*not* redundant — it makes every image contribute equally regardless of feature norm. The head has **zero
configuration**, so it cannot drift from the spec.

**Probe hyperparameters** are his suggested initial config (AdamW, lr 1e-3, wd 1e-4, batch 64, ≤200 epochs, checkpoint
on best validation accuracy). He permits adjusting them from validation results *provided the change is reported*, so
each is a config field and every run records what it used.

**Encoders.** ResNet-18 (512-d pre-classifier) on both datasets; DINOv2 ViT-S/14 (384-d class token, via timm) on one
dataset of our choice. **We chose Aircraft**, where ResNet-18 is weakest — that is where the encoder comparison carries
the most information. Each encoder owns its preprocessing: DINOv2 needs 518×518 (14×37), and ViT-S/14 cannot take a
size that is not a multiple of 14.

**Reporting.** Mean and sample std over three runs, **no confidence intervals** — three runs do not support them and
none were asked for. `report` refuses to emit a cell holding fewer runs than the protocol requires.

**Two incidents that changed the code.** A CLIP cache built with the wrong GELU variant survived the fix as "up to
date", so `model_name` and `weights_tag` now enter the cache key. And cuDNN TF32 (on by default for Ampere+) put a
3.1e-3 error floor on ResNet-18 features that varied with batch shape; extraction now disables it (6.2e-6 residual).

---

## 3. What is already built for Stage 2

The flow machinery is done and tested — only the head that *uses* it is missing.

- **Path + objective.** Linear conditional OT path, `L_CFM = E‖v_θ(x_t,t) − (x₁−x₀)‖²`. Simulation-free; asserted
  exactly zero when the field is exact.
- **Solver.** Fixed-step Euler and midpoint, measured convergence orders **1.0** and **2.0** against a closed-form
  solution. Step count is never chosen internally — it is an experiment axis. No `no_grad` anywhere, so one code path
  serves inference *and* rolled-out training. Ships with a `straightness` metric for later reflow work.
- **Velocity net.** MLP over `concat(x, sinusoidal_time_embedding(t))`. Time enters through an embedding, not a raw
  scalar — a field that ignores `t` still trains and still transports, so that failure would be silent.
- **2D toy — the real validation.** Our first toy (three round Gaussians) passed the acceptance criterion immediately
  and was **worthless**: nearest-prototype already scored 1.000 before transport. We rebuilt it with the geometry the
  Stage 1 diagnosis found — classes stretched along one shared direction. The flow then recovers the entire loss:
  **0.875 → 1.000** at stretch 5, **0.787 → 1.000** at stretch 7, and nothing on round classes (the control).

**Engineering state:** 240 tests, 90.8% coverage (gate at 85%), strict test-first history, atomic result writes, and
every run recording its git commit and exact subset indices. The test split is unreachable by construction — the loop
opens it only *after* `fit` returns.

---

## 4. Open questions — please have a view before the next meeting

1. **Whitening baseline first?** If the gap is anisotropy with roughly shared covariance, a single whitening may close
   much of it. Cheap to run, and we should not claim a flow is *needed* until we have that number.
2. **Unconditional vs class-conditional field?** The toy field is unconditional on purpose — at inference the class is
   unknown, so it must infer the basin from position. Whether that survives 100 classes in 384-d is genuinely open.
3. **Step count.** Is accuracy-vs-NFE a headline result or an appendix?
4. **Simulation-free vs rolled-out training.** Rolled-out is supported by the solver but is a heavy GPU job — Stage 2
   or Stage 3? And is reflow / straightening in scope at all, or a stretch goal?
5. **Probe tuning on DINOv2.** At full data it is 74.1% ±0.001, near-saturated. Tune from validation (and report it, as
   permitted), or keep his initial config everywhere for comparability?

---

## 5. What I need from you

- **A position on Q1 and Q2** — they change what we build next week, not just how we report it.
- **Review `config/stage1.yaml` and `config/figures.yaml`.** Those two files *are* the protocol; fixing a wrong choice
  now is far cheaper than after the write-up quotes the numbers.
- **Pick up Phase 8 with me** — the flow-matching head on real features. The `HeadContext` seam already exists for the
  tensors it will need, and the head interface will not change.

Reproduce everything in three commands: `uv sync && ./scripts/check`, then
`uv run python -m fm_fewshot sweep --config config/stage1.yaml`, then `uv run python -m fm_fewshot report`.

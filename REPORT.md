# Flow matching as a layer

CVLAB Summer Project 2026. Frozen encoders, small labelled subsets, and the
question of whether a flow-matching block can serve as a layer inside a
classifier rather than as a generative sampler.

Every number in this report is a row of `results/TABLE.md`, which is
regenerated from stored runs by `uv run python -m fm_fewshot report` and never
edited by hand. A row is the mean and sample standard deviation over the
protocol's three runs, and the three run directories under `results/` are named
`<timestamp>_<dataset>-<encoder>-<head>-k<K>-s<seed>`, so any number here can be
traced to the runs that produced it and to the config committed beside them.
Nothing is compared against published results; the protocol is the one the
project specified, and comparisons stay inside it.

---

## 1. What was asked, and what was built

Three stages, each specified in its own write-up, each treated as source of
truth over our own plan.

**Stage 1, baselines.** Classification on DTD (partition 1, 47 classes) and
FGVC-Aircraft (variant level, 100 classes), all classes, the official
train/val/test splits. Two baselines: a linear probe, and image-derived class
prototypes classified by cosine. Frozen ResNet-18 on both datasets, frozen
DINOv2 ViT-S/14 on Aircraft. K in {5, 10, full} training images per class,
three seeds, top-1 on the complete test split.

**Stage 2, FM as the last layer.** The classification head is replaced by an FM
block that transports features toward the Stage 1 image prototypes, and the
prediction is the cosine to the nearest prototype after T Euler steps. Both
training schemes are built and compared: standard flow matching, which fits the
velocity field without seeing the downstream loss, and rolled-out training,
which backpropagates an endpoint loss through all T solver steps.

**Stage 3, FM before a frozen linear classifier.** The Stage 1 linear probe is
trained first and frozen. An FM block is inserted ahead of it and initialized to
exact identity, so the untrained system reproduces the probe rather than
approximating it. Two strategies, and they are deliberately not Stage 2's two
schemes: end-to-end rolled-out cross-entropy through the whole rollout, and
standard FM on a coupling whose target is built from the classification gradient
and recomputed as training proceeds.

The grid is 156 Stage 1 and Stage 2 runs and 108 Stage 3 runs, all on
byte-identical training subsets. One evaluation loop runs every head, so the
claim "only the head changed" is structurally true rather than asserted.
`scripts/check_subsets` verifies it over the finished results and reports 264
runs with every head at each setting on identical rows.

---

## 2. Stage 1: the baselines

| Dataset / encoder | K | prototype | linear probe |
|---|---|---|---|
| DTD, ResNet-18 | 5 | **0.483** | 0.468 |
| | 10 | 0.517 | **0.526** |
| | full | 0.588 | **0.626** |
| Aircraft, ResNet-18 | 5 | 0.167 | **0.207** |
| | 10 | 0.191 | **0.266** |
| | full | 0.252 | **0.368** |
| Aircraft, DINOv2 | 5 | 0.255 | **0.430** |
| | 10 | 0.291 | **0.568** |
| | full | 0.362 | **0.741** |

**The heads swap places, and where they swap depends on the dataset.** On DTD at
K=5 the training-free prototype head wins, and the probe overtakes it by K=10.
On Aircraft the probe leads at every K including K=5. The low-shot advantage of
a training-free head is real but not universal: it appears on 47 texture classes
and not on 100 fine-grained ones, where class means sit too close together for a
nearest-centroid rule and a discriminative boundary pays off immediately.

**DINOv2 features are far more linearly separable than they are
prototype-friendly.** On Aircraft at full data the gap between the two heads is
0.379 for DINOv2 and 0.116 for ResNet-18. Switching encoder improved the probe
by 0.373 and the prototype head by only 0.110.

That gap has a single cause, and it matters for Stage 2. Whitening by the shared
within-class covariance and applying the identical centroid rule reaches 0.768,
matching or beating the probe in all nine cells. Outlier dimensions and an
off-origin cloud were both tested and rejected. DINOv2 places the classes well,
but its classes are not spherical, and a cosine centroid rule discards exactly
that structure. Since the specification makes those same prototypes the
transport target for Stage 2, the FM block is in effect being asked to learn the
whitening, and it has a floor at 0.362, a ceiling at 0.768 and a probe reference
at 0.741 to be read against.

**Nothing is data-saturated.** No cell plateaus between K=10 and full, and the
largest jumps are at the top end. The size curves are real curves, not three
points near an asymptote.

---

## 3. Stage 2: transport to the prototypes

Both schemes, T in {4, 12}, three cells, three K, three seeds. The baseline is
the prototype head, because that is what the block replaces.

**The layer helps on Aircraft and hurts on DTD, with no overlap.** It beats the
prototype baseline in all 24 Aircraft rows and loses in all 12 DTD rows. There
is no cell in between.

| Cell | best Stage 2 | prototype | probe |
|---|---|---|---|
| Aircraft, DINOv2, full | **0.637** (+0.275) | 0.362 | 0.741 |
| Aircraft, ResNet-18, full | 0.307 (+0.055) | 0.252 | 0.368 |
| DTD, ResNet-18, K=10 | 0.508 (-0.009) | 0.517 | 0.526 |

The headline gain of +0.275 is large, and it is worth being precise about what
it means. The block recovers a substantial part of the distance between the
prototype rule and the probe on the cell where that distance is widest, which is
the cell whose whitened-centroid reference is 0.768. It does not reach the
probe, on any cell.

**Standard training beats rolled-out training in 16 of 18 matched
comparisons.** The two exceptions are both Aircraft with DINOv2 at K=full, which
is the cell the headline number comes from and the least representative cell in
the study. Rolled-out training costs memory linear in T and, on this protocol,
usually buys nothing.

**The heads classify well without transporting features onto the prototypes.**
This is the geometric finding and it was not expected. After the full solve the
transported point is typically still far from its own class prototype: on
Aircraft with DINOv2 the solve closes about 95% of the starting distance while
finishing with a transported norm near 2 against a unit-norm target, and the
mean cosine to the correct prototype is 0.115. Pairwise cosine among transported
test points falls in all twelve representative runs, from 0.443 to between 0.006
and a small positive number, so the block spreads the cloud out rather than
collapsing it onto class targets. What the block learns is a direction, not a
destination. Read literally, "flow matching to class prototypes" is not what the
trained system does, even where it works.

**T = 12 beats T = 4 for both schemes, and the two gaps mean different things.**
For standard training it is one field integrated more finely, so the gap
measures curvature. For rolled-out training the two settings are two separately
trained models, so the gap measures capacity.

---

## 4. Stage 3: a block before the frozen probe

The baseline here is the linear probe, and the block starts as the identity, so
any movement is attributable to training rather than to initialization.

### The Main Comparison

At K=10 and T=12, the setting the specification asks to be presented:

| Cell | probe | Strategy 1 | Strategy 2 |
|---|---|---|---|
| DTD, ResNet-18 | 0.526 | 0.501 (**-0.025**) | 0.520 (-0.006) |
| Aircraft, DINOv2 | 0.568 | 0.570 (+0.002) | 0.567 (-0.002) |

**As specified, neither strategy beats the probe.** At T=12 no configuration
across the nine cells gains past its own seed spread, and every movement past
that spread is a loss. At T=4 two cells do gain past both their own spread and
the probe's: +0.006 on Aircraft with DINOv2 at K=10 and +0.005 on Aircraft with
ResNet-18 at K=full. Neither reaches a percentage point, and neither survives at
T=12.

The expectation was recorded before the runs: a gain at low K, little at
K=full. The K=full half held. The low-K gain, which is the question the stage
exists to ask, reached one cell out of six and only at 0.006. This is a
prediction that failed rather than a result read after the fact.

### The two strategies fail in two different ways

**Strategy 1 memorizes.** It drives its training cross-entropy to 1e-6 and makes
the frozen classifier's test margin worse. The loss curve keeps falling long
after validation accuracy has turned over, which is visible in the Stage 3
training panels and is why checkpoint selection on validation accuracy is a
protocol rule rather than an option.

**Strategy 2, read literally, diverges.** Its training loss peaks between 1e13
and 1e20, because the target is rebuilt from the field that is being trained
toward it and nothing bounds the total.

Both failures have a fix, and both fixes are things the specification raised and
left unchosen. They are in section 5.

### What the pictures show

The feature-space panels are undramatic at the graded settings: both strategies
move the cloud slightly and neither reorganizes it. Two things sharpen that.

Under t-SNE the before and after clouds are indistinguishable, while the PCA
panel on DTD with ResNet-18 clearly shows Strategy 1 inflating the cloud
outward. Same runs, two projections. The transport is largely radial, visible to
a linear projection and invisible to a neighbour embedding, and it does not
change which points are near which.

Every PCA panel states the share of the variance its two axes carry, which turns
"why did PCA read well on some cells" from an impression into a number. Two axes
carry between 19% and 29% on the Stage 3 cells. The one apparent exception,
50.6% on Aircraft with ResNet-18 in Stage 2, is an artifact: prototypes are unit
norm while raw features have norm near 49, so the leading axis is spent on the
scale gap. Projecting both onto the unit sphere first drops the same panel to
21.5% and separates the prototypes, which had looked piled on one point.

---

## 5. Beyond the specification

These are deviations. They are reported here and never in `TABLE.md`, and their
runs are stored separately from the graded grid.

**Strategy 1's regularizer works.** A displacement penalty at `lambda_disp`
0.01 takes DTD from 0.501 to 0.528 against a probe at 0.526. It does not start
winning, it stops losing, which is what a fix for memorization should look like:
the penalty discourages exactly the large representation change that did not
transfer. A velocity penalty does nothing either way.

**Constraining Strategy 2's target is the only thing that gains materially.**
Capping the target at `|| zhat' - z || <= rho || z ||`, with `rho` selected on
mean validation top-1 rather than on the test number it is reported by, picks
`rho = 0.15` and reaches **0.613 against the probe's 0.568** on Aircraft with
DINOv2, about eight times the probe's seed spread. On DTD it reaches 0.535
against 0.526, which is inside the probe's own spread and is not a gain we
claim.

Three reasons to believe the Aircraft number rather than treat it as one lucky
cell: the validation ranking reproduces the test ranking at every cap; the curve
has an interior optimum rather than a trend followed to the edge, since `rho`
approaching zero must return exactly to the probe; and the capped rows' seed
spread is 0.002 to 0.005, far smaller than the gain.

**Two schemes that were not asked for.** A hybrid of the two strategies matches
the better of its two parts and never exceeds them, so they are not
complements. Starting Stage 3 from a trained Stage 2 field recovers most of what
Strategy 1 loses on DTD, which fits the memorization reading, but it is not
competitive with the capped Strategy 2 and it gives up the property that made
the stage readable, namely the untrained system being the probe exactly.

**The remaining Strategy 2 knobs change nothing.** Ten configurations of step
size and target-improvement steps span 0.005 on validation, less than a single
cell's seed spread, and every one diverges in training.

---

## 6. What we cannot claim

**The DTD Stage 2 numbers are partly an artifact.** Those cells sit at the floor
of the checkpoint-selection grid, so their values are as much a property of the
selection rule as a measurement of the block.

**One gain is not a method.** The capped Strategy 2 result rests on one cell.
DTD gives the same configuration a gain inside the probe's spread, so the honest
statement is that the constraint helps where the block had room to help and does
nothing where it did not.

**The Stage 3 grid runs both T=4 and T=12** while the specification asks for a
single T chosen once and used throughout. The panels therefore draw four method
lines where one pair was requested. The evidence in them is that T changes very
little, which is arguably the case for picking one, but the choice has not been
made.

**Reverse flow is built and never run.** The module, the CLI entry point and its
tests are green, but no run has produced its output, so nothing in this report
rests on it.

---

## 7. Reproducing this

```bash
uv sync

# Feature caches, once per dataset, split and encoder.
uv run python -m fm_fewshot features --dataset dtd --split train --encoder resnet18

# The grids. --dry-run lists the runs without executing them.
uv run python -m fm_fewshot sweep --config config/stage1.yaml
uv run python -m fm_fewshot sweep --config config/stage2_standard.yaml
uv run python -m fm_fewshot sweep --config config/stage2_rolled.yaml
uv run python -m fm_fewshot sweep --config config/stage3_ce.yaml
uv run python -m fm_fewshot sweep --config config/stage3_guided.yaml

# The table and the figures, from stored results.
uv run python -m fm_fewshot report
uv run python -m fm_fewshot figures --config config/figures.yaml --stage 3
```

Encoders are frozen and features are cached once per dataset, split and
encoder, so a classifier run finishes in under a minute and every figure
redraws from stored results without a GPU. Subsets are drawn from a seeded
sampler shared by all heads, and subset selection and classifier initialization
use separate seed streams. Same seed and same device gives bit-identical
results; across devices, agreement within the reported standard deviation.

`uv run pytest` runs 810 tests at 96.73% coverage.

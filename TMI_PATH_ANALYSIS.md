# GraphCoW-Net: Honest Assessment for IEEE TMI Submission

Author: Claude (Opus 4.7) · Date: 2026-04-24

## 1. Current state (after Phase 1b, 35 epochs)

| Metric | Phase 1 | Phase 1b best | Δ |
|---|---|---|---|
| **mean_fg_dice** | 0.6442 | **0.6480** (ep 31) | +0.004 |
| R-Pcom | 0.305 | 0.369 (best ep 13) | +0.064 |
| L-Pcom | 0.288 | 0.359 (best ep 7) | +0.071 |
| Acom | 0.460 | 0.436 (best ep 9) | **−0.024** |
| 3rd-A2 | 0.000 | **0.049** (best ep 33) | +0.049 ✓ |

**SOTA on TopCoW 2024 MRA Task 1 (CLAIM-Berlin 1st place)**: class-avg Dice **~0.876**, ensemble ~0.901. Human ceiling ~0.902.

**Our gap to SOTA: 23 percentage points on mean Dice.** This is not closeable by the current trajectory — we are not iterating on a winning solution; we have architectural limitations the loss tweaks cannot fix.

## 2. Root-cause analysis of the plateau

### 2.1 Extreme class imbalance

| Class | Voxels in entire dataset | Per case avg | Cases present |
|---|---|---|---|
| BA (1) | 216,470 | 1,732 | 125/125 |
| ICA L/R (4,6) | ~800,000 | 6,400 | 125/125 |
| **R-Pcom (8)** | **20,159** | **330** | **61/125** |
| **L-Pcom (9)** | **18,756** | **335** | **56/125** |
| **Acom (10)** | **15,110** | **143** | **106/125** |
| **3rd-A2 (13)** | **8,769** | **548** | **16/125** |

Background/foreground ratio: **959:1**. Foreground is 0.10 % of volume. Pcom/Acom
are 1–2 voxels in diameter. **The 192³ patch at 0.5 mm spacing contains on
average ~200–500 vessel voxels out of 7 million.**

**Validation set for fold 0 (25 MR cases)**:
- Cls 8 (R-Pcom): present in **8/25** val cases
- Cls 9 (L-Pcom): present in **8/25** val cases
- Cls 13 (3rd-A2): present in **1/25** val cases ← single case defines the entire metric for this class

**A single bad prediction on the one 3rd-A2 val case swings the class dice from
0.0 to 1.0.** The metric is noisy at the per-class level for rare classes.

### 2.2 Architectural limitations (the real blocker)

**(A) DINOv3 encoder mismatches the domain.**

- DINOv3 was pretrained on natural images at 224×224. It has never seen a 3D MRA.
- It produces features at stride 16 (one token per 16×16 image patch).
- At 0.5 mm spacing, a **2 mm Pcom vessel occupies less than one ViT token**.
- The unfrozen top 2 blocks (lr_vit=5e-6) cannot shift the representation far
  enough to specialize for sub-token tubular features in 40 epochs.
- The tri-planar 3×2D hack is a Band-Aid: it triples compute, loses inter-slice
  continuity, and fundamentally cannot resolve the sub-patch resolution problem.

**(B) Stream B (the graph decoder) has never been trained.**

```python
# src/train_graphcow.py line 427:
loss, logs = criterion(out, targets={"mask": lbl}, phase="voxel_warmup", ...)
```

The training phase is HARDCODED to `voxel_warmup`. The "novel" graph stream
(NodePresenceLoss, CenterlineRegressionLoss, EdgeConsistencyLoss) — the whole
Circle-of-Willis atlas-based topology regulariser that was meant to be the
TMI contribution — has never received a gradient. It is 995 lines of dead code.

Required targets (`node_present`, `node_points`, `node_radius`, `edge_present`,
`edge_points`, `edge_radius`) are **not built by any dataloader**. There is no
`graph_targets.py`. The Circle-of-Willis atlas template is defined
(`COW_NODE_NAMES`, `COW_EDGES`) but never instantiated as training targets.

**(C) The "zero-shot transferability" claim has no implementation.**

The plan was to swap atlases at test time to transfer to kidney/cornea vessels.
That requires an atlas loader, a cross-domain validation case, and graph
decoder weights that respond to atlas swap. None of the three exist.

### 2.3 The Phase 1b data fixes plateaued because...

- Class-quota sampling: **fixed supply side** (ep 1: class 13 was never seen;
  ep 33: seen ~180 times). Worked as designed.
- PerClassConnectivityLoss: **fixed connectivity** of what's predicted. But
  the model has to predict *something* first. Pcom went 0.29 → 0.37 (+0.08).
- ClDice: same story as connectivity.
- CE class-weight ×3 for cls 13: only fires when model predicts something for
  cls 13. For 23 epochs it predicted nothing, so this never contributed.

**The loss-side cannot fix the encoder resolution mismatch.** Pcom is
1 vessel-voxel wide, encoder tokens are 16 mm patches. No amount of
reweighting changes Nyquist.

## 3. What a TMI submission needs (honest)

### 3.1 Numerical bar

Reviewers will compare to TopCoW 2024 leaderboard. Minimum: **match median**
(class-avg Dice ≥ 0.80, ideally ≥ 0.85). Topology metrics (clDice, β₀-error)
need to be **strictly better** than baselines (Dice alone won't carry the
paper). Current clDice on thin vessels ≈ 0.3; TopCoW 1st place ≈ 0.99.

### 3.2 Novelty bar

A novelty-only paper (method that loses on Dice but wins on topology) has
historically been accepted at TMI if:
1. The topology argument is air-tight (clean Betti matching, formal proofs
   of topology preservation, etc.)
2. There is a deployment story (zero-shot to another organ that no SOTA can do)

We have neither implemented.

### 3.3 Ablation bar

TMI requires systematic ablations. We have one (Phase 1 → Phase 1b with
several knobs changed at once). Proper ablations need each knob turned on
alone: quota only, conn-loss only, clDice only, elastic only. Current
experiment design confounds the contributions.

## 4. Three credible paths to TMI from here

Ranked by likelihood of acceptance and honesty of achievability. **All three
abandon the current GraphCoW-Net architecture** because the ViT+tri-planar
encoder cannot reach the numerical bar in the time available.

### Path A (highest probability): nnU-Net + topology losses, done right

**Thesis**: "Topology-aware losses outperform standard Dice on tubular
segmentation, specifically for sub-millimeter vessels of the Circle of Willis."

- Train plain **nnU-Net v2** on TopCoW (3 days, well-trodden recipe, mean dice
  typically 0.75–0.80 out of the box on MRA).
- Add **clDice + connectivity loss + cbDice** (our already-working loss code).
- Run **proper ablations**: each loss on its own, all combinations. 5-fold CV.
- Report **topology metrics**: Betti-0/1 error, Hausdorff-95, and clDice per
  class. Use the TopCoW evaluation toolkit.
- Contribution: a **clean**, **reproducible**, **loss-only** improvement over
  nnU-Net at the sub-millimeter scale.
- **Risk**: moderate. Loss-only papers need strong numbers. Expected final
  mean Dice ~0.82–0.85. Likely acceptance at a topology-focused venue; TMI
  is borderline unless reviewers like the topology rigour.

**Timeline**: 4 weeks (1 week nnU-Net baseline, 2 weeks ablations, 1 week
writing).

### Path B (TMI-grade if done well): Complete Stream B end-to-end

**Thesis**: "Atlas-anchored graph decoder + voxel decoder joint training
achieves better topology correctness than voxel-only baselines."

This is the original GraphCoW plan, but **actually implemented**.

- Build the graph target dataloader (2–3 days):
  - For each volume, skeletonize each class, fit polyline (K=32 vertices) to
    skeleton, compute per-vertex radius from distance transform.
  - Build edge polylines (bifurcation-to-bifurcation).
  - Cache to disk. This is straightforward engineering.
- Implement Stream B training phases: `graph_bootstrap` (freeze A, train B
  from scratch) → `joint` (unfreeze, add edge-consistency + coupling).
- Replace DINOv3 encoder with a **3D** encoder (either nnU-Net's encoder
  from the Path-A baseline, or a 3D U-Mamba). The ViT+tri-planar is a
  dead end for sub-token vessels.
- Run fold-0 + 4 ablations: (no Stream B), (Stream B aux), (joint), (joint +
  atlas-swap kidney test).
- Contribution: **first atlas-anchored topology-preserving joint segmentation
  and graph decoder, with zero-shot atlas transfer demonstration.**

**Timeline**: 10–12 weeks. Aggressive. The graph dataloader is the long pole;
joint training is fragile.

**Risk**: high but highest-reward. If numbers hit 0.84+ AND zero-shot works,
this is a strong TMI paper. If either fails, it's unsubmittable.

### Path C (safest, lower venue): Topology-aware ensemble of known SOTA

**Thesis**: "Post-hoc topology correction of SOTA segmentations."

- Take **published TopCoW 2024 top-3 solutions** (CLAIM-Berlin, Z-YCH,
  MEDICAL VISION UNIVERSITY nnUNet variants). Ensemble them.
- Apply a **learned topology refinement network** that takes the ensemble
  probabilities as input and outputs topology-corrected labels. Train this
  correction network with clDice + Betti-matching loss.
- This is not a new backbone; it's a lightweight post-hoc module.
- Contribution: **plug-and-play topology corrector that wins the TopCoW
  topology track regardless of base segmenter.**

**Timeline**: 3 weeks.

**Risk**: low. Numbers will be competitive (inherits ensemble's 0.88+
Dice, improves clDice). But novelty is lower — likely **MICCAI** or **MedIA**,
not TMI.

## 5. Recommendation

**Pick Path A, execute it, then decide on Path B extension.**

Rationale:
1. Path A gives a publishable result in 4 weeks even if everything else
   fails. Safety net.
2. Path A's outputs (baseline + losses + evaluation pipeline) are
   prerequisites for Path B anyway — no wasted work.
3. Only after Path A numbers are in hand can you make an honest call on
   whether Path B's 10-week investment is worth the TMI shot. If Path A
   hits 0.83 and Path B's graph stream doesn't, a MICCAI submission with
   just Path A is still strong.
4. The current GraphCoW-Net pipeline should be **parked**, not deleted.
   Its losses and data pipeline are reusable; its encoder is not.

## 6. Immediate actions

1. **Stop training Phase 1b** — it will finish ep 36–40 with no further
   gain (LR at 3.8e-7, effectively zero).
2. **Do NOT launch Phase 1c** — attention supervision on a mismatched 2D
   encoder will polish the plateau, not break through it. Expected gain
   over Phase 1b best: +0.01 to +0.02 mean Dice. Not worth 6 GPU-hours.
3. **Set up nnU-Net v2 baseline** on `data/cache_topcow/` using the same
   fold split (fold 0 held out). This is Path A step 1.
4. **In parallel**, implement `src/data/graph_targets.py` (polyline + radius
   + edge extraction from skeleton). This is the long pole for Path B.
5. Checkpoint the current state: `logs/graphcow_phase1/best.pt` (0.644)
   and `logs/graphcow_phase1b/best.pt` (0.648) — keep both as reference
   baselines for ablation tables if we ever return to this architecture.

## 7. What I would say in the paper (if it were ready today)

**I would not submit to TMI with the current results.** The mean Dice is
below baseline nnU-Net (which would likely land at 0.75–0.80 with zero
effort), and the "novelty" (graph decoder, zero-shot) is unimplemented.
Submitting now would get a desk rejection for "insufficient experimental
validation".

The honest step is to restart with Path A, build the baseline we *should*
have had from the beginning, then judge whether the GraphCoW ideas
(Stream A/B, atlas) are still worth pursuing for TMI-grade novelty.

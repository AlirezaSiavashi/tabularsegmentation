# GraphCoW-Net — Next-Steps Plan to Reach TMI Quality

Author: Claude (Sonnet 4.6) · Date: 2026-04-21 · Reviewer: Codex

Target: raise mean_fg_dice from the current **0.6442** plateau to **≥0.76** on
TopCoW 2024, fold 0, MR, publishable for IEEE TMI, while preserving the
zero-shot-transferable Stream A / Stream B factorisation.

------------------------------------------------------------------------
## 1. Diagnosis of the current plateau

Phase-1 (voxel_warmup) training (job 1116, 9h 12m, ep 101/120 when analysed):

| epoch | val mean_fg_dice |
|------:|------------------|
|   024 | 0.170 |
|   049 | 0.546 |
|   074 | 0.630 |
|   089 | 0.635 |
|   094 | 0.642 |
|   099 | **0.6442 ← best** |

**Per-class val dice at ep 099 (the real signal):**

| cls | name        | dice   | status                          |
|----:|-------------|--------|---------------------------------|
|   1 | BA          | 0.856  | ok                              |
|   2 | R-PCA       | 0.813  | ok                              |
|   3 | L-PCA       | 0.832  | ok                              |
|   4 | R-ICA       | 0.902  | strong                          |
|   5 | R-MCA       | 0.752  | ok                              |
|   6 | L-ICA       | 0.905  | strong                          |
|   7 | L-MCA       | 0.774  | ok                              |
|   8 | R-Pcom      | **0.305** | **FAIL — tiny vessel**       |
|   9 | L-Pcom      | **0.288** | **FAIL — tiny vessel**       |
|  10 | Acom        | **0.460** | **WEAK — tiny midline vessel** |
|  11 | R-ACA       | 0.733  | ok                              |
|  12 | L-ACA       | 0.755  | ok                              |
|  13 | 3rd-A2 (remapped from 15) | **0.000** | **FAIL — not learned at all** |

Mean over classes 1–13 = 0.644. **The main-vessel branch is already at strong
TMI-class performance (0.75–0.90). The plateau is driven almost entirely by
three rare classes (Pcom L/R, Acom) and by 3rd-A2 which is never learned.**

Root causes, from most to least impactful:

1. **3rd-A2 (class 13) has 0.0 dice.** This class is present in only
   ~16/125 cases. With fold 0 and MR-only we get ~2 val cases with it and a
   handful of train cases. fg_prob=0.7 picks a random present class weighted by
   `1/sqrt(count)`, so 3rd-A2 is rarely the crop centre. Also, the model must
   *co-exist* 3rd-A2 with Acom/A2 in the same neighbourhood; a simple remap to
   label 13 without extra supervision gives it no chance.
2. **Pcom L/R are 1–2 voxel thick** at 0.5 mm isotropic and about 15 mm long.
   Standard Dice does not give them enough gradient mass, and clDice was
   disabled during Phase 1 to avoid OOM. Re-enabling clDice (with
   gradient-checkpointed skeletonisation) is the single biggest expected win.
3. **LR has decayed to 1.2 % of peak** (cosine with 120 epochs) while the
   network has not yet converged on rare classes. The network is effectively
   frozen during the late epochs with no further signal.
4. **Only 250 sample patches / epoch × 100 cases** — each case is seen
   ~2.5×/epoch. Small classes rarely appear in 192³ patches (even with
   inverse-freq fg sampling) because their bounding box is small.
5. **No elastic deformation.** Thin tubular structures generalise better with
   elastic + bias-field augmentations, which are currently off.

The cb loss has already saturated at ~0.11 (flat from ep ~50). That term has
done its job — the bulk of remaining gain is **topology on the skeleton** +
**class-aware sampling** + **LR restart**, not more cbDice.

------------------------------------------------------------------------
## 2. Roadmap (three phases, each gated by Codex review)

```
        ┌──────────────────────────────────────────────────────────────┐
        │   Phase 1          Phase 1b           Phase 2         Phase 3 │
        │   voxel_warmup  →  fine-tune     →   graph_bs    →   joint    │
        │   (running)        (this plan)      (Stream B)      (unify)   │
        │   0.64 ✓           0.70 target      0.74 target    0.78 target │
        └──────────────────────────────────────────────────────────────┘
```

Phase 1 finishes naturally at ep 120 in ~90 min from the time of writing.
**Do not kill it.** Let it produce `logs/graphcow_phase1/best.pt` and
`train_log.jsonl` complete. Everything below starts from that checkpoint.

------------------------------------------------------------------------
## 3. Phase 1b — voxel fine-tune with re-enabled topology and class-aware sampling

Goal: close the gap on Pcom/Acom and **unlock class 13 (3rd-A2)**.
Expected: mean_fg_dice **0.64 → 0.70** (+0.06).

### 3.1 Changes in order of expected Δdice

| # | Change | Expected Δ | Rationale |
|---|--------|-----------:|-----------|
| A | Re-enable clDice with iters=5, weight=0.5 | +0.02 | Was disabled due to OOM; now we have headroom (14.3 / 40 GB). |
| B | Introduce per-class **sampling-quota** (guarantee ≥1 patch/epoch for cls 8,9,10,13) | +0.02 | Current inverse-freq sampling under-samples ultra-rare classes. |
| C | LR warm-restart to `lr_dec=5e-5`, short cosine to 5e-6 over **40 epochs** | +0.015 | Current lr is ~1.5e-6; model has stopped learning. |
| D | Elastic deformation (sigma=4, alpha=80) + bias-field in-place of pure rotation 50 % of the time | +0.01 | Thin vessel generalisation. |
| E | Raise `--samples-per-epoch 250 → 400` (still fits wall-time budget; 400 × 23 s = ~2.5 h/ep with 40 epochs = 100 h < 192 h limit) | +0.005 | More gradient updates per case. |

**Do NOT change:** `unfreeze_top_k=2`, `lr_vit=1e-5`, class-weight boost
`{8,9,10}=1.5`, LR-flip + relabel aug, patch=192³. These all work.

**Memory test first.** Before submitting the 40-epoch run, run a 2-epoch
dry-run with A+B+D active to measure peak GPU mem. If it exceeds 36 GB, cut
`samples-per-epoch` back to 300 or reduce cld iters to 4.

### 3.2 Required code changes (small, localised)

| file | change |
|------|--------|
| `src/data/topcow_dataset.py` | Add `class_quota` list parameter to `TopCoWDataset`; when set, every call to `__getitem__` that hits the quota yields a crop centred on that class instead of uniform. Quota iterates round-robin. |
| `src/data/topcow_dataset.py` | Add optional elastic deform in `_augment` (guarded by `elastic_prob=0.3`). Use existing `F.grid_sample` + random displacement field; fall back to identity if input is too small. |
| `src/train_graphcow.py` | Add `--resume-weights-only` (loads model state but restarts optimiser + scheduler). Add `--class-quota "8,9,10,13"` CLI. Add `--elastic-prob`. |
| `submit_graphcow_phase1b.sh` | New SLURM script, see §3.3. |

No changes to `GraphCoWNet`, `graphcow_losses.py` (cld-iters is already a CLI
flag), or `graphcow_encoder.py`. All changes are additive — Phase 1 resume
path remains identical for reproducibility.

### 3.3 Phase 1b submission script (to be written)

```bash
#!/bin/bash
#SBATCH --job-name=gc_ph1b
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_ph1b_%j.log
#SBATCH --error=logs/slurm_ph1b_%j.log
...
$PYTHON src/train_graphcow.py \
    --cases data/cache_topcow/cases.json \
    --variant vits16 --unfreeze-top-k 2 --modalities mr --fold 0 \
    --patch 192 192 192 --batch-size 1 --grad-accum 2 \
    --epochs 40 --samples-per-epoch 400 --num-workers 4 --num-classes 14 \
    --lr-vit 5e-6 --lr-encoder-adapter 2e-5 \
    --lr-decoder 5e-5 --lr-graph 1e-4 --weight-decay 1e-4 \
    --warmup-epochs 2 --val-every 2 \
    --cld-iters 5 --cb-iters 4 --cld-weight 0.5 --cb-weight 0.5 \
    --class-quota 8,9,10,13 --elastic-prob 0.3 \
    --resume logs/graphcow_phase1/best.pt --resume-weights-only \
    --out logs/graphcow_phase1b
```

Notes:
* `val-every 2` instead of 5 so we catch the gain earlier and can early-stop.
* LRs halved vs Phase 1 peak (warm restart, not cold restart).
* 40 epochs × ~5 min = ~3.5 h training; well within budget.

------------------------------------------------------------------------
## 4. Phase 2 — Graph bootstrap (Stream B only)

Goal: activate the graph decoder and get structured node/edge predictions that
match the CoW atlas (13 nodes, 13 edges), **without degrading the voxel Dice**.
Expected: +0.02 to +0.04 on mean_fg_dice via edge-consistency regularisation
later in Phase 3. Stream B itself evaluated on **node-presence F1** and
**centerline Chamfer distance** (new TMI metrics — not just Dice).

### 4.1 Freezing strategy

* **Freeze**: DINOv3 encoder (incl. the 2 unfrozen blocks), tri-planar adapter,
  Stream A voxel decoder.
* **Train**: graph decoder (cross-attention blocks, node/edge heads, radius
  head, presence head).
* Why: Stream B is initialised from scratch; letting it pull gradient through
  A would damage the 0.70 A-stream we just built. Standard "warmup the new
  head against frozen backbone" trick.

### 4.2 Loss weights (graph_bootstrap)

Set `phase="graph_bootstrap"` in `criterion(...)`. This already zeros the
voxel losses. Keep graph weights at defaults:
`node_presence=1.0, centerline=2.0, radius=0.5, prelim_graph=0.3`.

### 4.3 Data targets

`graph_bootstrap` needs `node_present`, `node_points`, `node_radius`,
`edge_present`, `edge_points`, `edge_radius`. These are **built from the atlas
JSON**, not from the voxel labels directly. The atlas loader must:

1. Produce a fixed 13-node, 13-edge template in normalised [0,1]³ patch coords.
2. Per training patch: run a cheap connected-components on the cropped label
   volume, detect which atlas nodes are present, fit a polyline of fixed
   length K=32 to the skeleton of each edge, compute radius via Euclidean
   distance transform.

This loader does not exist yet. It is the single biggest Phase-2 engineering
task. See §4.5.

### 4.4 Submission script `submit_graphcow_phase2.sh`

```bash
...
--epochs 30 --samples-per-epoch 300 \
--lr-vit 0 --lr-encoder-adapter 0 --lr-decoder 0 --lr-graph 5e-4 \
--resume logs/graphcow_phase1b/best.pt --resume-weights-only \
--phase graph_bootstrap \
--out logs/graphcow_phase2
```

`--lr-* = 0` is the cheap way to freeze A-stream without touching
`trainable_parameter_groups`. Alternative: add `--freeze-voxel` flag that sets
`.requires_grad=False` on Stream A params; preferred for cleanliness.

### 4.5 New code required

| file | purpose |
|------|---------|
| `src/data/graph_targets.py` (NEW) | Build per-patch node/edge targets from label + atlas. |
| `src/data/topcow_dataset.py` | Optional `return_graph=True`; when set, also return the graph-target dict. |
| `src/train_graphcow.py` | Add `--phase {voxel_warmup,graph_bootstrap,joint}` CLI, pass through to `criterion(..., phase=phase)`. Currently hardcoded `"voxel_warmup"`. Add `--freeze-voxel`. |
| `configs/atlas_cow_13.json` (NEW) | The 13-node CoW template with canonical positions and edge list. Version-control this — it's the "organ-specific" knob that we swap at zero-shot time. |

### 4.6 Evaluation additions

* `val/node_presence_f1` per class.
* `val/centerline_chamfer_mm` aggregated over present edges.
* Still log mean_fg_dice (should stay flat at ~0.70 since A is frozen).

------------------------------------------------------------------------
## 5. Phase 3 — Joint training (Streams A + B together)

Goal: use graph predictions to **fix topology errors in Stream A** via
`EdgeConsistencyLoss` and `GraphVoxelCouplingLoss` (both already implemented
in `graphcow_losses.py`, already gated on `phase="joint"`). Expected:
mean_fg_dice **0.70 → 0.78+**.

### 5.1 Unfreezing schedule

* Epoch 0–5: unfreeze decoder only, LR low.
* Epoch 6–20: unfreeze tri-planar adapter, keep ViT frozen at lr 0.
* Epoch 20–40: unfreeze top 2 ViT blocks again at `lr_vit=2e-6`.
* Epoch 40–60: everything at shared cosine decay to 1e-6.

This is a staged unfreeze borrowed from the SwinUNETR-TMI recipe: it stops the
fresh gradients from the graph side from blowing up the A-stream that we
invested 40+120 epochs in.

### 5.2 Loss weights (joint)

```
dice_ce=1.0  cb=0.5  cld=0.5  betti=0.0 (keep off: too expensive)
node_presence=0.5  centerline=1.0  radius=0.25
edge_consistency=1.0  coupling=0.5
deep_sup=0.5  prelim_graph=0.2
```

The **halving** of node_presence / centerline / radius from Phase 2 is
intentional: in joint phase the graph side is there to *regularise* the voxel
side, not to dominate it.

### 5.3 Submission script `submit_graphcow_phase3.sh` (60 epochs)

```bash
... --phase joint --epochs 60 --samples-per-epoch 400 \
--lr-vit 2e-6 --lr-encoder-adapter 1e-5 \
--lr-decoder 2e-5 --lr-graph 2e-5 \
--resume logs/graphcow_phase2/best.pt --resume-weights-only ...
```

### 5.4 Evaluation additions

* **Betti-number error** (post-hoc, not in loss): count components, loops on
  binarised prediction vs GT, average error. This is **the TMI topology
  metric**.
* **HD95** per class.
* **Zero-shot transfer probe**: run Phase-3 checkpoint on one unseen
  structure (e.g. kidney vessels from KiPA22) with only the atlas swapped.
  Frozen. Report Dice. Positive number → headline result for TMI.

------------------------------------------------------------------------
## 6. Milestones and exit criteria

| phase | wall time | checkpoint | mean_fg_dice target | stop condition |
|-------|----------:|------------|--------------------:|----------------|
| 1     | done (~10 h) | `logs/graphcow_phase1/best.pt` | 0.64 | ep 120 / natural finish |
| 1b    | ~4 h      | `logs/graphcow_phase1b/best.pt` | **0.70** | no +0.005 over 6 epochs |
| 2     | ~4 h      | `logs/graphcow_phase2/best.pt` | 0.70 (A stable) + F1 ≥0.6 (B) | graph F1 plateau |
| 3     | ~8 h      | `logs/graphcow_phase3/best.pt` | **0.78+** | no +0.003 over 10 epochs |

If Phase 1b misses 0.70 by more than 0.015, revisit the class-quota sampler
and the 3rd-A2 relabel strategy before touching Phase 2 (garbage-in /
garbage-out for the graph side otherwise).

------------------------------------------------------------------------
## 7. Risks and mitigations

1. **clDice OOM returns with quota + elastic aug.** Mitigation: dry-run 2
   epochs, fall back to cld_iters=4 or cld_weight=0.3.
2. **Quota sampler starves common classes.** Mitigation: quota is only 1 of
   every N=4 samples (25 %), rest uses existing sampler.
3. **Graph-target builder is slow.** Mitigation: pre-compute per-case graph
   targets at the full volume, then crop-map to patch at dataloader time.
   Same pattern used by TopCoW 2024 1st-place.
4. **Phase 3 collapses A-stream.** Mitigation: hard constraint — if
   `val_mean_fg_dice` drops > 0.02 below Phase-1b best for 3 consecutive vals,
   abort and reduce graph loss weights by ×0.5.
5. **Zero-shot transfer doesn't work.** Mitigation: it's a stretch goal, not
   a gate; report whichever organ scales best. Atlas quality dominates.

------------------------------------------------------------------------
## 8. Concrete deliverables before next Codex review

1. `NEXT_STEPS_PLAN.md` — this document.
2. Wait for job 1116 to finish naturally (ETA: ~90 min from writing).
3. Draft + stage (not yet submit) `submit_graphcow_phase1b.sh`.
4. Draft `src/data/topcow_dataset.py` diff for class-quota and elastic aug.
5. Draft `src/train_graphcow.py` diff for `--resume-weights-only`,
   `--class-quota`, `--elastic-prob`, `--phase`, `--freeze-voxel`.
6. Submit Phase-1b only after Codex sign-off on 1, 3–5.

Phase 2 and Phase 3 code is described here at the design level; concrete diffs
come after Phase 1b lands.

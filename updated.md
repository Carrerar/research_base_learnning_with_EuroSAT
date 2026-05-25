# Research Protocol Decision Log — EuroSAT SSL Project

Date: 2026-05-25

Project pipeline:
Read → Reproduce → Refine → Research → Report

Current status:
- Reproduce: DONE
- Refine: DONE
- Research: preparing protocol freeze

---

# Context

During Refine phase, the project used:

- Official split:
  - Train: 21600
  - Val: 2700
  - Test: 2700
- Ratio:
  - 80 / 10 / 10
- Goal:
  - Reproduce paper baseline (~98.57%)
  - Refine supervised multispectral pipeline

Refine conclusion:
- T1-R2 fixed (13-band raw + AdamW) is official supervised winner
- EuroSAT supervised classification is close to saturation
- Remaining research value likely lies in:
  - low-label regime
  - representation learning
  - SSL

Research phase (RQ1) was inspired by:
- Original EuroSAT paper showing ~75% accuracy when training with only 10% labeled data

This created uncertainty:

Question:
Should the entire project now switch to a 10/90 split to match the paper's low-label benchmark?

---

# Final Research Decision

## DO NOT switch to literal 10/90 train/test split

Reason:
Modern SSL and representation learning research does NOT usually change the global dataset split.

Instead:
- keep validation fixed
- keep test fixed
- vary ONLY the amount of labeled training data

The original paper's 10/90 protocol is considered outdated by modern standards because:
- insufficient validation separation
- overly large test set
- poor compatibility with modern SSL evaluation
- difficult fair comparison across methods

---

# Official Protocol Freeze (FINAL)

## Dataset split (fixed forever)

Train: 21600
Val: 2700
Test: 2700

Ratio:
80 / 10 / 10

This split is now frozen and shared across:
- Refine
- Research
- Report

DO NOT regenerate new global splits.

---

# Low-Label Research Protocol

Research phase studies:
"How much labeled data is needed?"

NOT:
"How should the dataset be split globally?"

Therefore:

Only subsample from TRAIN split.

Val and Test remain unchanged for ALL experiments.

---

# Label Fraction Protocol (Official)

Create stratified subsets from TRAIN ONLY:

| Fraction | Samples |
|----------|----------|
| 1%       | 216      |
| 5%       | 1080     |
| 10%      | 2160     |
| 100%     | 21600    |

Requirements:
- stratified sampling
- fixed seed (42)
- nested subsets preferred:
  - 1% ⊂ 5% ⊂ 10% ⊂ 100%

This is now the official Research evaluation protocol.

---

# Important Clarification

The original paper's "~75% at 10%" result is still scientifically useful.

However:
- we reproduce the "low-label regime concept"
- NOT the literal 10/90 train/test split

The meaningful variable is:
- amount of labeled training data

NOT:
- exact train/test ratio

---

# Relationship Between Refine and Research

Refine and Research answer DIFFERENT questions.

## Refine question

"Can multispectral inputs and better methodology improve supervised EuroSAT classification?"

This justified:
- full training data
- stable tuning
- 80/10/10 split

---

## Research question

"Can SSL improve representation quality and reduce dependence on labeled data?"

This justifies:
- low-label evaluation
- label fractions
- fixed validation/test protocol

Therefore:
Refine and Research DO NOT need identical train-fraction protocols.

---

# Critical Decision

## DO NOT rerun entire Refine under label fractions

Reason:
- computationally expensive
- low scientific return
- combinatorial explosion
- Refine already completed its purpose

Instead:
carry forward ONLY the official supervised winner.

---

# Official Supervised Baseline (Frozen)

Condition B:

- Input: 13-band raw Sentinel-2
- Model: ResNet-50
- Init: ImageNet pretrained
- Optimizer: AdamW
- Recipe: T1-R2 fixed
- Metrics:
  - Accuracy
  - Macro-F1
  - Hard-class F1

This becomes the official supervised baseline for all Research comparisons.

---

# Immediate Next Steps Before SSL

## Step 1 — Protocol Freeze
Freeze:
- splits
- metrics
- training recipe
- seed policy
- evaluation methodology

No more supervised tuning.

---

## Step 2 — Generate Low-Label Supervised Baseline Curve

Run Condition B under:
- 1%
- 5%
- 10%
- 100%

This creates the official supervised low-label baseline.

Purpose:
measure how ImageNet-pretrained supervised learning degrades as labels decrease.

---

## Step 3 — Verify Low-Label Stability

Check:
- subset generation
- stratification
- variance across seeds
- training stability
- metric reliability

---

## Step 4 — Tiny SSL Pilot

Before full SSL pretraining:
run small-scale sanity tests.

Verify:
- SSL loss decreases
- embeddings are non-trivial
- fine-tuning works
- data pipeline is correct
- memory usage is stable

DO NOT immediately launch full 100–200 epoch SSL runs.

---

# SSL Research Direction (RQ1)

Main question:

"Does SSL pretraining on Sentinel-2 unlabeled data improve low-label EuroSAT classification?"

Primary comparison:
- SSL vs ImageNet

Secondary comparison:
- SSL vs Scratch

Main evaluation focus:
- low-label regime (1% / 5% / 10%)

NOT:
- squeezing tiny gains at 100%

---

# Key Research Philosophy

The project is transitioning from:

"benchmark optimization"

to:

"representation learning analysis"

The objective is no longer:
- maximizing supervised accuracy at all costs

The new objective is:
- understanding data efficiency
- understanding representation quality
- understanding transfer behavior in low-label settings

---

# Final Guidance For Claude Code

When implementing future experiments:

1. NEVER regenerate global dataset splits.
2. NEVER switch to literal 10/90 train/test evaluation.
3. ALWAYS keep:
   - val fixed
   - test fixed
4. ONLY subsample from TRAIN split.
5. Use stratified subsets with seed 42.
6. Treat T1-R2 fixed as the official supervised baseline.
7. Refine phase is COMPLETE and should not be reopened unless a critical methodological bug is discovered.
8. Research phase starts with low-label supervised baselines BEFORE SSL.
9. SSL experiments must isolate the effect of representation learning, not optimizer tuning.
10. Maintain scientific continuity between Refine and Research.



## RESEARCH PHASE UPDATE
SSL Strategy (OFFICIAL)

Research phase được chia thành 2 tầng rõ ràng để tránh confounders và giữ scientific clarity.

---

### Phase RQ1-A — Controlled SSL Study (OFFICIAL CORE)

Mục tiêu:

- isolate representation learning effect
- giữ continuity với Refine
- giảm confounders
- tạo clean baseline comparison

Official backbone:

- ResNet-50

Reason:

Nếu vừa:
- đổi backbone
- đổi architecture family
- đổi SSL objective

thì sẽ không biết performance gain đến từ:
- SSL
- hay architecture.

Do đó:

Research core phase PHẢI giữ cùng backbone với Refine.

---

### Official Comparison Matrix

| Condition | Backbone | Pretraining |
|---|---|---|
| A | ResNet-50 | Scratch |
| B | ResNet-50 | ImageNet |
| C | ResNet-50 | SSL |

Variable chính cần isolate:

- representation learning quality

KHÔNG phải:

- architecture scaling
- transformer advantage
- foundation model scaling

---

### Recommended SSL Methods For RQ1-A

Ưu tiên:

- SimCLR
- MoCo-v2
- BYOL

Reason:

- dễ compare với supervised baseline
- clean experimental design
- low engineering complexity
- phù hợp cho controlled study

---

### Important Clarification About MAE

MAE KHÔNG bị loại bỏ.

Ngược lại:

- MAE / SatMAE được xem là hướng rất mạnh cho multispectral remote sensing
- đặc biệt phù hợp với Sentinel-2 spectral-spatial redundancy

Tuy nhiên:

MAE thường kéo theo:

- ViT-based architecture
- patch tokenization
- masking strategy
- decoder design
- optimization changes

=> đây không còn là:

"same model + different pretraining"

mà gần như là:

"new representation paradigm"

Nếu dùng MAE ngay từ đầu:

- confounders tăng mạnh
- khó isolate SSL effect
- khó compare trực tiếp với Refine baseline

Do đó:

MAE KHÔNG phải official starting point của Research phase.

---

### Phase RQ1-B — MAE / Foundation-Model Extension (OPTIONAL)

Chỉ thực hiện nếu:

- RQ1-A hoàn tất
- baseline low-label curves đã ổn định
- còn compute / thời gian

Lúc này project mới mở rộng sang:

- architecture + SSL synergy
- foundation-model style learning
- masked spectral-spatial reconstruction

Possible models:

- MAE
- SatMAE
- DINOv2
- Prithvi
- SatDINO
- ViT-Small

Potential research question:

> Does masked spectral-spatial reconstruction outperform contrastive SSL for multispectral low-label learning?

---

### Why MAE Is Promising For Remote Sensing

Sentinel-2 có:

- strong spatial redundancy
- strong spectral redundancy
- multispectral correlation

Ví dụ:

- SWIR ↔ NIR correlations
- vegetation structure shared across bands
- masked spectral content often inferable from context

=> reconstruction objective của MAE đặc biệt phù hợp.

Ngoài ra:

contrastive SSL trong remote sensing thường khó vì:

- augmentation design dễ phá spectral meaning
- color jitter có thể không physically valid
- aggressive crops có thể làm mất semantic land-cover context

Trong khi MAE:

- ít phụ thuộc augmentation mạnh
- học reconstruction thay vì view invariance
- phù hợp multispectral physics hơn

Do đó:

MAE/SatMAE được xem là high-value extension direction sau khi controlled SSL baseline đã được establish.

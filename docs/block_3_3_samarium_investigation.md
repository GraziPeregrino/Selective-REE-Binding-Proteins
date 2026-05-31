# Block 3.3: Samarium Investigation

## Question

Block 3.1's per-element Spearman breakdown identified Samarium as an
outlier: Sm Spearman was 0.04, compared with a 0.49-0.78 range for the
other 14 REEs in MOESM3. Block 3.2's randomized hyperparameter search
confirmed that tuning did not resolve the issue. Across 250
trial-by-outer-fold measurements, mean Sm Spearman was -0.135, with a
range of -0.62 to +0.43.

This investigation examines why Sm is difficult to rank.

## Finding

The evidence indicates that target saturation is a major contributor to
poor Samarium ranking performance.

MOESM3 reports normalized_logD: each variant's REE preference is scaled
relative to its strongest REE. LanM orthologs frequently exhibit peak or
near-peak normalized values for Sm.

| REE | Mean | Std | % at >= 0.95 |
|---|---:|---:|---:|
| Lanthanum | 0.5130 | 0.1863 | 0.0% |
| Cerium | 0.7647 | 0.1188 | 1.4% |
| Praseodymium | 0.8916 | 0.0848 | 29.3% |
| Neodymium | 0.9331 | 0.0589 | 48.4% |
| **Samarium** | **0.9651** | **0.0725** | **81.5%** |
| Europium | 0.8270 | 0.0843 | 5.1% |
| Gadolinium | 0.5902 | 0.1429 | 0.8% |
| Terbium | 0.4897 | 0.1765 | 0.0% |
| Dysprosium | 0.3932 | 0.2038 | 0.0% |
| Holmium | 0.3132 | 0.2231 | 0.0% |
| Erbium | 0.2744 | 0.2341 | 0.0% |
| Thulium | 0.2558 | 0.2435 | 0.0% |
| Ytterbium | 0.2587 | 0.2639 | 0.2% |
| Lutetium | 0.2524 | 0.2806 | 1.4% |
| Yttrium | 0.2492 | 0.2435 | 0.0% |

Of 492 training orthologs, 44.1% have Sm normalized_logD exactly equal
to 1.0 and 81.5% have Sm values greater than or equal to 0.95. The
median is 0.9919. Values from the 75th percentile onward equal 1.0.

Spearman correlation requires meaningful rank information. Extensive
ties and near-ties sharply reduce the recoverable ranking signal.

## Mechanism

MOESM3 reports each variant's logD profile normalized relative to its
maximum-logD REE per replicate. When Sm is a preferred REE, its
normalized value approaches or reaches 1.0 regardless of differences in
absolute Sm behavior.

This compression removes distinctions among many variants. The current
feature representation may still have limitations, but the normalized
target itself restricts the ranking signal available to the model.

## Evidence

### 1. Distribution Evidence

Training-set Sm values show unusually severe saturation:

- 81.5% are greater than or equal to 0.95.
- 44.1% are exactly 1.0.
- The median is 0.9919.

Europium, Sm's immediate heavier neighbor, shows only 5.1% saturation.

### 2. Orthogonal Targeted Validation

Diep et al. 2026 MOESM6 reports targeted triplicate normalized_logD
profiles for four reference orthologs:

| Variant | Sm normalized logD mean | SD | n |
|---|---:|---:|---:|
| o-36 | 1.0000 | 0.0000 | 3 |
| o-127 | 1.0000 | 0.0000 | 3 |
| o-412 | 0.9076 | 0.0312 | 3 |
| Mex-LanM (o-621) | 1.0000 | 0.0000 | 3 |

Three of four reference profiles reach the Sm ceiling. These targeted
measurements support the biological plausibility of the saturation
pattern. They do not, by themselves, exclude all assay-related effects.

### 3. Exploratory Recovery Analysis

Removing saturated test observations with target values greater than or
equal to 0.95 yields:

| Subset | n | Sm Spearman |
|---|---:|---:|
| Full Sm test set | 124 | +0.121 |
| Non-saturated subset | 23 | +0.376 |

Sm ranking performance improves substantially on the unsaturated
subset, although the remaining sample is small and the resulting
correlation remains modest. This is an explanatory diagnostic, not a
replacement headline metric.

## Broader Per-Element Pattern

Across all 15 REEs, training-set saturation rate is inversely associated
with held-out per-element Spearman:

```text
Pearson r = -0.5364
p = 0.039
```
Saturation alone explains roughly 29% of per-element performance
variance. The relationship is not deterministic because two other
effects modulate it:

- **Heavy lanthanides (Tb-Lu) and Y**: Low family-wide preference
  (mean 0.25-0.49, std 0.20+), 0% saturation. Plenty of signal to
  rank, uniform Spearman in [0.50, 0.60].
- **Light lanthanides at peak (Pr-Eu)**: High family-wide preference,
  moderate-to-high saturation. Mixed performance.
- **Samarium**: At the peak of the family's preference distribution.
  Compression dominates. Spearman collapses unless saturated cases
  are excluded.

## Implications

### For this project
The Sm "outlier" should be reported as a documented dataset artifact,
not a model deficiency. Block 4 Streamlit deployment notes should
explicitly state that the model's confidence for Samarium predictions
is reduced due to target compression.

### For the field
A future iteration could rescore Sm rankings against absolute logD
rather than per-variant normalized values, when the original paper's
raw data become accessible. This would test whether absolute Sm
binding (rather than relative preference) is more rank-predictable.

### Not justified by this investigation
- Adding phylogenetic features to "fix" Sm
- Re-running hyperparameter search against this same test set
- Adjusting the model to over-weight Sm-related features

The model is well-calibrated. The target is not.

## Methodology Notes

All diagnostics were run on training data only (Diagnostic A's
denominator) or on the held-out test set after the tuned model was
finalized (Diagnostic A's numerator, Diagnostic B). No model
retraining occurred. No test-set decisions were made based on these
findings.

The two MOESM6 reference values are independent of our training
data; we did not retrain the model after seeing them.

## Conclusion

Block 3.3 is closed with a non-modeling finding: Samarium's poor
ranking performance is a target distribution artifact (81.5%
saturation in training, 44.1% at exact maximum), confirmed by
independent triplicate measurements in MOESM6 and by recovery to
0.376 Spearman on the non-saturated test subset. The model's
mechanism is appropriate; the dataset compresses Sm signal.
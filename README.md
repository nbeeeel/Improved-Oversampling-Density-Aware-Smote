# Density-Aware SMOTE

![PyPI version](https://img.shields.io/pypi/v/density-aware-smote?color=blue)
![Python versions](https://img.shields.io/pypi/pyversions/density-aware-smote)
![License](https://img.shields.io/github/license/nbeeeel/Improved-Oversampling-Density-Aware-Smote)

A Python package implementing **Density-Aware SMOTE** for intelligent oversampling in imbalanced classification tasks. Unlike standard SMOTE, this method allocates synthetic samples **proportionally to local minority density**: fewer synthetic points in sparse/unsafe zones, more in dense/safe zones. This reduces noise amplification and preserves true minority structure.

---

## Features

* **Density-aware allocation** of synthetic samples per minority instance
* **Flexible sampling strategy**: `'auto'`, a ratio (`0<r≤1`), dict of target counts, or callable
* **Configurable neighbor policy**: `'nearest'`, `'random'`, `'farthest'`
* **scikit-learn / imbalanced-learn compatible**: `fit_resample(X, y)`, works in pipelines
* **Reproducible**: `random_state`, `n_jobs` for parallel kNN
* **Optional plotting helpers** for class balance and synthetic placements

---

## Installation

```bash
pip install density-aware-smote
```

---

## Why Density-Aware?

Plain SMOTE distributes new samples uniformly across minority instances. In highly imbalanced or overlapping datasets, this can:

* Over-synthesize **border/noisy** regions, increasing misclassification
* Under-represent **well-formed** minority clusters

**Density-Aware SMOTE** computes a local density score $\rho_i$ (e.g., inverse of mean kNN distance) for each minority sample $x_i$, then allocates new samples per-point using a normalized weight:

$$
w_i \propto \left(\frac{1}{\bar{d}_k(x_i)+\epsilon}\right)^{\alpha}
\qquad\Rightarrow\qquad
n_i = \text{round}\!\left(R \cdot \frac{w_i}{\sum_j w_j}\right)
$$

* $\bar{d}_k(x_i)$: average distance to k nearest minority neighbors
* $\alpha \ge 0$: density emphasis (higher = more bias to dense/safe areas)
* $R$: total number of synthetic points to create

Synthetic points follow the standard SMOTE interpolation:

$$
x_{\text{new}} = x_i + \lambda \cdot (x_{nn} - x_i),\quad \lambda \sim \mathcal{U}(0,1)
$$

---

## API Overview

```python
from density_aware_smote import DensityAwareSMOTE

sampler = DensityAwareSMOTE(
    sampling_strategy='auto',   # or float, dict, callable
    k_neighbors=5,              # kNN for density & interpolation
    alpha=1.0,                  # density emphasis
    neighbor_selection='nearest', # 'nearest' | 'random' | 'farthest'
    density_metric='knn',       # 'knn' (default)
    n_jobs=None,                # parallelism for kNN
    random_state=42
)

X_res, y_res = sampler.fit_resample(X, y)
```

### Parameters

* **sampling_strategy**:
  * `'auto'`: upsample minority to majority count
  * `float r (0<r≤1)`: make minority = `r * majority`
  * `dict`: `{class_label: desired_count}`
  * `callable(y) -> dict`
* **k_neighbors**: k for both density computation and neighbor interpolation
* **alpha**: density exponent; `0` ≈ uniform SMOTE, `>1` strongly favors dense regions
* **neighbor_selection**:
  * `'nearest'`: classic SMOTE neighbor selection
  * `'random'`: random neighbor from kNN set (increases diversity)
  * `'farthest'`: longest kNN edge (expands cluster support cautiously)
* **density_metric**: `'knn'` uses inverse mean kNN distance
* **random_state**: seed for reproducibility
* **n_jobs**: parallel jobs for kNN computations

### Attributes

* **sampling_strategy_**: resolved sampling plan after `fit_resample`
* **classes_**, **class_distribution_**: discovered during fitting

---

## Quick Start Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from density_aware_smote import DensityAwareSMOTE

# Create imbalanced dataset
X, y = make_classification(n_samples=2000, n_features=20, weights=[0.95, 0.05],
                           n_informative=6, class_sep=1.0, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Apply Density-Aware SMOTE
sampler = DensityAwareSMOTE(sampling_strategy=0.5, k_neighbors=7, alpha=1.0, random_state=0)
X_res, y_res = sampler.fit_resample(X_train, y_train)

# Train classifier on resampled data
clf = RandomForestClassifier(random_state=0).fit(X_res, y_res)
print(classification_report(y_test, clf.predict(X_test), digits=4))
```

---

## Pipeline Integration

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from density_aware_smote import DensityAwareSMOTE

pipe = Pipeline(steps=[
    ('scale', StandardScaler(with_mean=False)),
    ('smote', DensityAwareSMOTE(sampling_strategy='auto', k_neighbors=5, alpha=0.8, random_state=42)),
    ('clf', LogisticRegression(max_iter=200))
])

scores = cross_val_score(pipe, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
print("F1-macro (cv=5):", scores.mean())
```

---

## Advanced Usage

### Custom Sampling Strategy

```python
# Specific target counts per class
sampler = DensityAwareSMOTE(
    sampling_strategy={
        0: 1000,  # Class 0: keep existing count
        1: 800,   # Class 1: oversample to 800 samples
        2: 600    # Class 2: oversample to 600 samples
    },
    k_neighbors=5,
    alpha=1.2,
    random_state=42
)
```

### Tuning Density Parameters

```python
# Higher alpha for stronger density bias
sampler = DensityAwareSMOTE(
    sampling_strategy='auto',
    k_neighbors=7,
    alpha=1.5,  # Strong bias toward dense regions
    neighbor_selection='random',  # Increase diversity
    random_state=42
)
```

---

## Visualization

```python
from density_aware_smote.viz import plot_class_balance, plot_synthetic_2d

# Before and after class distribution
plot_class_balance(y, title="Before Oversampling")
plot_class_balance(y_res, title="After Density-Aware SMOTE")

# For 2D data: visualize synthetic sample placement
plot_synthetic_2d(X, y, X_res, y_res, title="Synthetic Sample Placement")
```

---

## Practical Guidelines

### When to Use
* Minority clusters exist but are overwhelmed by majority class
* Borderline noise or class overlap is present
* Standard SMOTE creates too many samples in unsafe regions
* Need to preserve minority class structure while oversampling

### Parameter Tuning
* **k_neighbors**: 5–10 is common; larger values for smoother data manifolds
* **alpha**: 
  * `0.5–1.0`: moderate density bias
  * `1.0–2.0`: strong bias toward dense regions
  * `0`: equivalent to uniform SMOTE
* **neighbor_selection**:
  * `'nearest'`: classic SMOTE behavior
  * `'random'`: reduces sampling bias, increases diversity
  * `'farthest'`: conservative expansion of cluster support

### Preprocessing Considerations
* **Scaling**: standardize features when distances are scale-dependent
* **High dimensions**: consider dimensionality reduction (PCA) first
* **Categorical features**: encode appropriately; assumes numeric features
* **Extreme imbalance**: use incremental ratios with validation

---

## Comparison with Other Methods

| Method | Allocation Strategy | Focus | Best For |
|--------|-------------------|--------|----------|
| **SMOTE** | Uniform per minority instance | Boundary interpolation | Moderate imbalance |
| **Borderline-SMOTE** | Focus on borderline instances | Hard examples | Clear class boundaries |
| **ADASYN** | More samples in hard regions | Difficult examples | Complex decision boundaries |
| **Density-Aware SMOTE** | Proportional to local density | Safe/dense regions | Noisy boundaries, cluster preservation |

---

## Complexity

* **Time**: $O(n \log n)$ for kNN search with tree-based indices
* **Space**: $O(n)$ for storing density weights and neighbor indices
* **Sample generation**: Linear in number of synthetic points $R$

---

## Known Limitations

* Distance-based density can be unreliable in very high dimensions
* Extremely scarce minority classes (< 10 samples) may still overfit
* Requires appropriate distance metrics for mixed data types
* Performance depends on the quality of local density estimation

---

## Contributing

Issues and pull requests are welcome. Please include:
* Minimal reproducible example
* Dataset characteristics
* Environment details (`python`, `numpy`, `scikit-learn`, `imbalanced-learn` versions)

---

## References

* N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," *JAIR*, 2002
* H. He et al., "ADASYN: Adaptive Synthetic Sampling Approach," *IJCNN*, 2008
* H. Han et al., "Borderline-SMOTE: A New Over-Sampling Method," *ICIC*, 2005

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{scalogram_cnn_smote_2023,
  title={A Scalogram-based CNN Ensemble Method with Density-Aware SMOTE Oversampling for Improving Bearing Fault Diagnosis},
  author={[Author Names]},
  journal={IEEE Access},
  volume={PP},
  number={99},
  pages={1--1},
  year={2023},
  doi={10.1109/ACCESS.2023.3332243}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

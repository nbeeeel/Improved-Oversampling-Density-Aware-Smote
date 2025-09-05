from density_aware_smote.smote import DensityAwareSMOTE
from density_aware_smote.visualization import plot_class_distribution, plot_synthetic_samples
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, weights=[0.9, 0.1], random_state=42)

smote = DensityAwareSMOTE(k_neighbors=3, density_exponent=0.7, random_state=42)
X_res, y_res = smote.fit_resample(X, y)

plot_class_distribution(y, y_res)
plot_synthetic_samples(X, y, X_res, y_res)

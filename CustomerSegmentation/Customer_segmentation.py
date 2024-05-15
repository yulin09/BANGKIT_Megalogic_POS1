import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from extract_customersdata import customers_df
import numpy as np

# Assuming customers_df is already loaded
# Extract features for clustering
features = customers_df[['previous_purchase', 'total_spend']]

# Convert DataFrame columns to NumPy arrays
previous_purchase = features['previous_purchase'].values
total_spend = features['total_spend'].values

# Normalize the features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Apply the weights 70:30
weighted_features = features_scaled * [0.3, 0.7]

# Fit KMeans algorithm
algorithm = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan')
algorithm.fit(weighted_features)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

# Generate mesh grid for plotting decision boundaries
h = 0.05  # Adjust the step size to reduce memory usage and increase the number of points
x_min, x_max = weighted_features[:, 0].min() - 0.1, weighted_features[:, 0].max() + 0.1
y_min, y_max = weighted_features[:, 1].min() - 0.1, weighted_features[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot decision boundaries and clusters
plt.figure(1, figsize=(15, 7))
plt.clf()
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', linewidths=1, alpha=0.5)
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

plt.scatter(x=weighted_features[:, 0], y=weighted_features[:, 1], c=labels1, s=200)
plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], s=200, c='red', alpha=0.5, marker="X")
plt.ylabel('Weighted Total Spend')
plt.xlabel('Weighted Previous Purchase')
plt.title('Customer Segmentation with 70:30 Weighting on Total Spend and Previous Purchase')
plt.show()


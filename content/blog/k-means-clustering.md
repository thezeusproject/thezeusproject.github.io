<!-- `made by` https://cneuralnets.netlify.app/ -->
---
title: K-Means Clustering
subtitle: Most fundamental idea of unsupervised learning, i.e, what can you do when you don't have any labels to learn from? You just bucket "similar" points together! The "K" here stands for the number of clusters or buckets that the points are being segregated into.
author: shivanirudh
---

## What do we mean by similar?

![image](/assets/images/KMeans-intro.png)

For most cases, we treat similarity as a "closeness" metric, as in, how close two points are compared to a third point. For example, points $A$ and $B$ are "closer" and hence similar to each other than with points $C$ and $D$. Likewise, points $C$ and $D$ are "closer" and hence similar to each other.


## Approach

Since we have multiple points that are a part of each cluster, the argument can be what if point C is closer to A, but equidistant from B and D, which "cluster" would it belong to? To handle this case, we represent each cluster as the mean point or centroid of all the points belonging to that cluster. Therefore, the points would belong to the cluster whose centroid they are closest to. 

Mathematically, we want to minimize the distance(variance) within the points belonging to each cluster:

\[
\text{arg min}_S \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2
\]

- \( S_i \): Set of points in cluster \( i \)  
- \( \mu_i \): Mean (centroid) of cluster \( i \) and is computed as:

\[
\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x
\]

## Algorithm

1. **Initialize**: Choose K initial centroids randomly  
2. **Assignment Step**: Assign each point to the nearest centroid  
3. **Update Step**: Recalculate the centroid of each cluster  
4. **Repeat**: Until centroids don't change (convergence)

![K-means_convergence.gif](/assets/images/K-means_convergence.gif)

## Issues with K-Means clustering

- Initialization Sensitivity: Random centroids can lead to poor convergence   
- Need to specify K: You must know or guess the correct number of clusters
- Assumes Spherical Clusters: Doesn't work well for non-globular shapes
- Sensitive to Outliers: Outliers can heavily skew centroids 

## Choosing the right "K"

### Elbow method
![K-Means-elbow](/assets/images/KMeans-Elbow-Method.png)

- Plot **inertia vs. K**
- Look for the "elbow" where adding clusters doesn’t reduce loss much

### Silhouette Score

\[
\text{Silhouette} = \frac{b - a}{\max(a, b)}
\]

- \( a \): Mean distance to points in the same cluster  
- \( b \): Mean distance to points in the nearest cluster  
- Range: [-1, 1] → Higher is better

## Python code from scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

K = 4
max_iters = 100

np.random.seed(42)
random_indices = np.random.choice(len(X), size=K, replace=False)
centroids = X[random_indices]


def compute_distances(X, centroids):
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

for i in range(max_iters):
    distances = compute_distances(X, centroids)
    labels = np.argmin(distances, axis=1)
    
    new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
    
    if np.allclose(centroids, new_centroids):
        break
    
    centroids = new_centroids

colors = ['r', 'g', 'b', 'y']
for k in range(K):
    plt.scatter(X[labels == k][:, 0], X[labels == k][:, 1], s=30, color=colors[k], label=f'Cluster {k}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X', label='Centroids')
plt.title("K-Means Clustering (from scratch)")
plt.legend()
plt.show()
```



# Ternary Clustering

**Ternary Clustering** implements clustering algorithms for ternary data in {-1, 0, +1} — providing K-means adapted for ternary spaces, DBSCAN with ternary Hamming distance, hierarchical agglomerative clustering, and cluster validity indices (silhouette score, Davies-Bouldin index).

## Why It Matters

Clustering discovers structure in unlabeled data. For ternary-encoded fleet data (agent states, strategy vectors, sensor readings), traditional Euclidean clustering assumes continuous values — but ternary data is discrete with only 3^D possible distinct points in D dimensions. Ternary Clustering exploits this discreteness: ternary K-means uses exact centroid rounding, DBSCAN uses Hamming distance for efficient neighbor search, and the small alphabet enables exhaustive enumeration for small D. Applications: fleet agent profiling, anomaly cluster detection, strategy group identification.

## How It Works

### Distance Metrics

**Squared Euclidean** (for K-means):
```
d²(a, b) = Σ (a_i - b_i)²   where a_i, b_i ∈ {-1, 0, +1}
```
Possible values per dimension: 0 (same), 1 (adjacent), 4 (opposite). Range: [0, 4D].

**Hamming** (for DBSCAN):
```
d_H(a, b) = |{i : a_i ≠ b_i}|
```
Range: [0, D]. More natural for discrete ternary data.

Both: **O(D)** per comparison.

### K-means Adapted for Ternary

Standard K-means with ternary-specific centroid computation:

```
1. Initialize k centroids randomly from data points
2. Assign each point to nearest centroid: O(N·K·D)
3. Recompute centroids: mean each dimension, round to nearest ternary
4. Repeat until centroids stable or max_iterations

Centroid rounding:
    mean < -0.5 → -1
    mean > 0.5  → +1
    else        → 0
```

Convergence: typically **O(I · N · K · D)** for I iterations.

### DBSCAN

Density-based clustering with Hamming distance:

```
For each point p:
    neighbors = points within ε Hamming distance
    if |neighbors| ≥ min_pts:
        p is a core point → expand cluster
    else:
        p is noise or border
```

Neighbor search: **O(N²)** naively, **O(N · log N)** with indexing. Cluster expansion: **O(N)** per point.

### Hierarchical (Agglomerative)

Bottom-up merging:

```
1. Start with N singleton clusters
2. Merge two closest clusters (single/complete/average linkage)
3. Repeat until 1 cluster or distance threshold

Linkage: single (min distance), complete (max), average (mean)
```

Cost: **O(N³)** for full dendrogram (N-1 merges, each O(N²) distance update).

### Validity Indices

**Silhouette Score** (per point):
```
s_i = (b_i - a_i) / max(a_i, b_i)
```
where a_i = mean intra-cluster distance, b_i = mean nearest-cluster distance. Range: [-1, 1]. Overall: mean of s_i.

**Davies-Bouldin**:
```
DB = (1/k) Σ max_{j≠i} ((σ_i + σ_j) / d(c_i, c_j))
```
Lower is better. Both: **O(N²)** to compute.

## Quick Start

```rust
use ternary_clustering::{Ternary, kmeans, nearest_centroid};

let data = vec![
    vec![Ternary::Pos, Ternary::Pos, Ternary::Neg],
    vec![Ternary::Pos, Ternary::Zero, Ternary::Neg],
    vec![Ternary::Neg, Ternary::Neg, Ternary::Pos],
    vec![Ternary::Neg, Ternary::Neg, Ternary::Zero],
];

let centroids = kmeans(&data, 2, 10);
for (i, c) in centroids.iter().enumerate() {
    println!("Cluster {} centroid: {:?}", i, c);
}
```

## API

| Function | Complexity | Description |
|----------|------------|-------------|
| `kmeans(data, k, iters)` | O(I·N·K·D) | K-means with ternary rounding |
| `dbscan(data, eps, min_pts)` | O(N²) | Density-based clustering |
| `hierarchical(data, linkage)` | O(N³) | Agglomerative dendrogram |
| `silhouette_score(data, labels)` | O(N²) | Cluster validity [-1, 1] |
| `davies_bouldin(data, labels)` | O(N²) | Cluster validity (lower better) |
| `nearest_centroid(point, centroids)` | O(K·D) | Find nearest cluster |
| `hamming_distance(a, b)` | O(D) | Count differing positions |

## Architecture Notes

Ternary Clustering provides the unsupervised learning layer for fleet agent profiling in SuperInstance. In γ + η = C, clusters of +1-heavy agents represent γ (growth-oriented subpopulations), clusters of -1-heavy agents represent η (avoidance-oriented), and mixed clusters with many 0s represent the neutral/uncommitted population. Integrates with `outlier-score` for anomaly detection and `ternary-bayesian` for probabilistic cluster assignment.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for fleet analytics architecture.

## References

1. MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations." *Berkeley Symposium on Mathematical Statistics and Probability*.
2. Ester, M. et al. (1996). "A Density-Based Algorithm for Discovering Clusters." *KDD*.
3. Rousseeuw, P. J. (1987). "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis." *Computational and Applied Mathematics*, 20, 53–65.

## License

MIT

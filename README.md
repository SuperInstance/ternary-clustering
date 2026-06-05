# ternary-clustering

Clustering algorithms for ternary-valued data vectors (`-1`, `0`, `+1`).

## Why This Exists

Most clustering libraries assume continuous or binary data. But many real-world systems produce ternary signals — sentiment (negative/neutral/positive), market direction (down/flat/up), sensor states (low/normal/high), or trimmed gene expression. Standard K-means and DBSCAN lose structural information when you encode ternary values as floats. This crate provides clustering algorithms that respect ternary geometry from the ground up: Hamming distance for DBSCAN, ternary-space centroid computation for K-means, and proper cluster validity indices that understand ternary distances.

## Core Concepts

- **`Ternary`** — An enum with three variants: `Neg` (-1), `Zero` (0), `Pos` (+1). All data is represented as vectors of these values.
- **Hamming distance** — Counts positions where two ternary vectors differ. The natural distance metric for discrete ternary data.
- **Ternary centroid** — The mean of a cluster's ternary vectors, rounded back to ternary values via thresholding at ±0.5.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-clustering = "0.1"
```

```rust
use ternary_clustering::*;

fn main() {
    // Build a dataset of ternary vectors
    let data = vec![
        vec![Ternary::Pos, Ternary::Pos, Ternary::Pos],
        vec![Ternary::Pos, Ternary::Pos, Ternary::Zero],
        vec![Ternary::Neg, Ternary::Neg, Ternary::Neg],
        vec![Ternary::Neg, Ternary::Neg, Ternary::Zero],
    ];

    // K-means with k=2
    let (assignments, centroids) = kmeans(&data, 2, 100);
    println!("Assignments: {:?}", assignments);
    println!("Centroids: {:?}", centroids);

    // DBSCAN clustering
    let labels = dbscan(&data, 1, 2);
    println!("DBSCAN labels: {:?}", labels);

    // Evaluate with silhouette score
    let score = silhouette_score(&data, &assignments);
    println!("Silhouette score: {:.4}", score);
}
```

## API Overview

### Distance Functions
- `ternary_sq_distance(a, b)` — Squared Euclidean distance between ternary vectors
- `hamming_distance(a, b)` — Number of differing positions

### Clustering Algorithms
- `kmeans(data, k, max_iters)` — K-means adapted for ternary spaces. Returns cluster assignments and centroids.
- `dbscan(data, eps, min_pts)` — Density-based clustering using Hamming distance. Returns `Option<usize>` labels (`None` = noise).
- `hierarchical(data, linkage)` — Agglomerative clustering. Returns a dendrogram as merge steps `(cluster_a, cluster_b, distance)`.

### Linkage Criteria
- `Linkage::Single` — Minimum inter-cluster distance
- `Linkage::Complete` — Maximum inter-cluster distance
- `Linkage::Average` — Average inter-cluster distance

### Cluster Validity
- `silhouette_score(data, assignments)` — Mean silhouette coefficient (−1 to +1; higher is better)
- `davies_bouldin_index(data, assignments, centroids)` — Davies-Bouldin index (≥0; lower is better)

### Utilities
- `nearest_centroid(point, centroids)` — Index of the closest centroid
- `compute_centroid(points)` — Ternary centroid of a point set

## How It Works

**K-means** initializes centroids by selecting evenly-spaced data points, then alternates between assigning each point to its nearest centroid (using squared Euclidean distance) and recomputing centroids by averaging member vectors and rounding back to ternary.

**DBSCAN** pre-computes all pairwise Hamming distances to find ε-neighborhoods, then expands clusters from core points (those with ≥ `min_pts` neighbors). Points not reachable from any core point are labeled as noise.

**Hierarchical clustering** starts with each point as its own cluster, then iteratively merges the closest pair according to the chosen linkage criterion. Pairwise Hamming distances are pre-computed once; cluster distances are derived from these.

**Silhouette score** computes for each point the ratio of inter-cluster distance to intra-cluster distance. The mean across all points measures overall clustering quality.

**Davies-Bouldin index** averages, across all clusters, the worst-case similarity ratio between each cluster and its most similar neighbor.

## Use Cases

1. **Sentiment analysis grouping** — Cluster documents represented as ternary sentiment vectors (negative/neutral/positive per feature) into opinion groups.
2. **Financial signal classification** — Group market indicators (down/flat/up per metric) to identify regime patterns.
3. **Genomic data clustering** — Cluster gene expression profiles after ternarization (underexpressed/normal/overexpressed) to find co-regulated gene groups.

## Ecosystem

- [`ternary-projection`](https://github.com/user/ternary-projection) — Dimensionality reduction for ternary data (PCA, t-SNE, random projection)
- [`ternary-graph`](https://github.com/user/ternary-graph) — Graph algorithms on ternary-weighted edges
- [`ternary-streaming`](https://github.com/user/ternary-streaming) — Streaming processing for ternary signals

## License

MIT

## See Also
- **ternary-pca** — related
- **ternary-projection** — related
- **ternary-topology** — related
- **ternary-graph** — related
- **ternary-network** — related


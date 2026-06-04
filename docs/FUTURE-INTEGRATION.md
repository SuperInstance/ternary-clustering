# Future Integration: ternary-clustering

## Current State
Implements clustering algorithms for ternary vectors: K-means with ternary centroids, DBSCAN with Hamming distance, hierarchical agglomerative clustering, and validity indices (silhouette score, Davies-Bouldin index).

## Integration Opportunities

### With ternary-cell / room-as-codespace
Group rooms by similarity. `kmeans()` clusters rooms into k groups based on their ternary state vectors. Rooms in the same cluster share ensign assignments and control policies. `dbscan()` discovers room clusters of arbitrary shape — identifying zones (kitchen cluster, storage cluster, corridor cluster) without pre-specifying k. The silhouette score validates whether the discovered clusters are meaningful.

### With ternary-hash
`TernaryMinHash` sketches approximate Hamming distance between room states. Replace `ternary_sq_distance()` in kmeans with MinHash-based distance for faster convergence on large datasets. The sketch-based approach scales to thousands of rooms.

### With ternary-trees
Decision tree leaf nodes define interpretable clusters. Each leaf is a cluster defined by a conjunction of ternary feature thresholds. Hierarchical clustering produces a dendrogram that IS a decision tree. Unify: use trees for interpretable clustering, k-means for fast centroid-based clustering, DBSCAN for density-based discovery.

## Potential in Mature Systems
In PLATO, clustering runs continuously at Layer 2. As rooms stream state updates, an incremental k-means maintains cluster assignments in real-time. When a room drifts from its cluster centroid (measured by Hamming distance), it's flagged for review. The Davies-Bouldin index monitors cluster quality — if it degrades, the system re-clusters with a different k. Clustering results propagate to Layer 1 as room group metadata for coordinated control.

## Cross-Pollination Ideas
**Music × Clustering:** Cluster ternary chord sequences by harmonic content. `kmeans()` discovers prototypical chord progressions (e.g., "ii-V-I cluster," "blues cluster"). `dbscan()` finds rare progressions as outliers. Silhouette scores identify the most distinct harmonic idioms. This creates a data-driven taxonomy of ternary music. Connects to `ternary-music`.

**Evolution × Clustering:** In `evolution-ternary`, species emerge from clustering organisms by genotype similarity. `dbscan()` with appropriate epsilon defines species boundaries naturally — dense genotype clusters are species, isolated individuals are new species candidates.

## Dependencies for Next Steps
- Incremental/streaming k-means for real-time room clustering
- Integration with `ternary-hash` for approximate clustering at scale
- Cluster stability metrics: how much do assignments change between ticks?

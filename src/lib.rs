//! # ternary-clustering
//!
//! Clustering algorithms for ternary data represented as vectors of `Ternary` values
//! (`-1`, `0`, `+1`). Provides K-means adapted for ternary spaces, DBSCAN with ternary
//! Hamming distance, hierarchical (agglomerative) clustering, and cluster validity
//! indices (silhouette score, Davies-Bouldin index).

#![forbid(unsafe_code)]

/// A ternary value: Negative (-1), Zero (0), or Positive (+1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    /// Convert to i8 representation.
    pub fn to_i8(self) -> i8 {
        match self {
            Ternary::Neg => -1,
            Ternary::Zero => 0,
            Ternary::Pos => 1,
        }
    }

    /// Convert from i8, clamping to {-1, 0, 1}.
    pub fn from_i8(v: i8) -> Self {
        if v < 0 {
            Ternary::Neg
        } else if v == 0 {
            Ternary::Zero
        } else {
            Ternary::Pos
        }
    }
}

/// Compute the squared Euclidean distance between two ternary vectors.
pub fn ternary_sq_distance(a: &[Ternary], b: &[Ternary]) -> i64 {
    a.iter().zip(b.iter()).map(|(x, y)| {
        let d = x.to_i8() as i64 - y.to_i8() as i64;
        d * d
    }).sum()
}

/// Compute Hamming distance between two ternary vectors (count of differing positions).
pub fn hamming_distance(a: &[Ternary], b: &[Ternary]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Compute the nearest ternary centroid for a given point.
/// Returns the index of the closest centroid.
pub fn nearest_centroid(point: &[Ternary], centroids: &[Vec<Ternary>]) -> usize {
    centroids.iter().enumerate()
        .min_by_key(|(_, c)| ternary_sq_distance(point, c))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Round a floating-point vector to the nearest ternary vector.
fn round_to_ternary(v: &[f64]) -> Vec<Ternary> {
    v.iter().map(|&x| {
        if x < -0.5 { Ternary::Neg }
        else if x > 0.5 { Ternary::Pos }
        else { Ternary::Zero }
    }).collect()
}

/// Compute the mean of a set of ternary vectors, then round to ternary.
pub fn compute_centroid(points: &[Vec<Ternary>]) -> Vec<Ternary> {
    if points.is_empty() {
        return Vec::new();
    }
    let dim = points[0].len();
    let n = points.len() as f64;
    let mut means = vec![0.0f64; dim];
    for p in points {
        for (i, val) in p.iter().enumerate() {
            means[i] += val.to_i8() as f64;
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }
    round_to_ternary(&means)
}

/// K-means clustering adapted for ternary data.
///
/// Returns cluster assignments (one per data point) and the computed centroids.
pub fn kmeans(data: &[Vec<Ternary>], k: usize, max_iters: usize) -> (Vec<usize>, Vec<Vec<Ternary>>) {
    if data.is_empty() || k == 0 {
        return (Vec::new(), Vec::new());
    }
    let k = k.min(data.len());
    let dim = data[0].len();

    // Initialize centroids by picking evenly spaced data points
    let mut centroids: Vec<Vec<Ternary>> = (0..k).map(|i| {
        data[i * data.len() / k].clone()
    }).collect();

    let mut assignments = vec![0usize; data.len()];

    for _ in 0..max_iters {
        // Assign each point to nearest centroid
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let new_assign = nearest_centroid(point, &centroids);
            if assignments[i] != new_assign {
                assignments[i] = new_assign;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids
        for c in 0..k {
            let members: Vec<&Vec<Ternary>> = data.iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == c)
                .map(|(_, p)| p)
                .collect();
            if !members.is_empty() {
                centroids[c] = compute_centroid(&members.iter().map(|&m| m.clone()).collect::<Vec<_>>());
            }
        }
    }

    (assignments, centroids)
}

/// DBSCAN clustering for ternary data using Hamming distance.
///
/// Returns cluster assignments where `None` indicates noise.
pub fn dbscan(data: &[Vec<Ternary>], eps: usize, min_pts: usize) -> Vec<Option<usize>> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let mut labels: Vec<Option<usize>> = vec![None; n];
    let mut cluster_id = 0usize;

    // Pre-compute neighborhoods
    let neighborhoods: Vec<Vec<usize>> = (0..n).map(|i| {
        (0..n).filter(|&j| hamming_distance(&data[i], &data[j]) <= eps).collect()
    }).collect();

    for i in 0..n {
        if labels[i].is_some() {
            continue;
        }

        let neighbors = &neighborhoods[i];
        if neighbors.len() < min_pts {
            continue; // noise for now
        }

        // Start new cluster
        labels[i] = Some(cluster_id);
        let mut seeds: Vec<usize> = neighbors.iter().filter(|&&j| j != i).copied().collect();
        let mut idx = 0;

        while idx < seeds.len() {
            let q = seeds[idx];
            if labels[q].is_none() {
                labels[q] = Some(cluster_id);
            }
            idx += 1;

            let q_neighbors = &neighborhoods[q];
            if q_neighbors.len() >= min_pts {
                for &nb in q_neighbors {
                    if labels[nb].is_none() {
                        labels[nb] = Some(cluster_id);
                        seeds.push(nb);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    labels
}

/// Linkage criterion for hierarchical clustering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Linkage {
    /// Minimum distance between clusters (single linkage).
    Single,
    /// Maximum distance between clusters (complete linkage).
    Complete,
    /// Average distance between clusters (average linkage).
    Average,
}

/// Agglomerative hierarchical clustering for ternary data.
///
/// Returns a dendrogram as a sequence of merge steps: (cluster_a, cluster_b, distance).
pub fn hierarchical(data: &[Vec<Ternary>], linkage: Linkage) -> Vec<(usize, usize, f64)> {
    let n = data.len();
    if n <= 1 {
        return Vec::new();
    }

    // Pre-compute pairwise distances
    let mut dist = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d = hamming_distance(&data[i], &data[j]) as f64;
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Track active clusters
    let mut active: Vec<bool> = vec![true; n];
    let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut cluster_label: Vec<usize> = (0..n).collect();
    let mut merges = Vec::new();
    let mut next_label = n;

    for _ in 0..(n - 1) {
        // Find closest pair of active clusters
        let mut best_dist = f64::MAX;
        let mut best_i = 0;
        let mut best_j = 1;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let d = cluster_distance(&cluster_members[i], &cluster_members[j], &dist, linkage);
                if d < best_dist {
                    best_dist = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        merges.push((cluster_label[best_i], cluster_label[best_j], best_dist));

        // Merge j into i
        let to_extend: Vec<usize> = cluster_members[best_j].clone();
        cluster_members[best_i].extend(to_extend);
        cluster_label[best_i] = next_label;
        next_label += 1;
        active[best_j] = false;
    }

    merges
}

fn cluster_distance(a: &[usize], b: &[usize], dist: &[Vec<f64>], linkage: Linkage) -> f64 {
    match linkage {
        Linkage::Single => {
            a.iter().flat_map(|&i| b.iter().map(move |&j| dist[i][j])).fold(f64::MAX, f64::min)
        }
        Linkage::Complete => {
            a.iter().flat_map(|&i| b.iter().map(move |&j| dist[i][j])).fold(f64::MIN, f64::max)
        }
        Linkage::Average => {
            let total: f64 = a.iter().flat_map(|&i| b.iter().map(move |&j| dist[i][j])).sum();
            total / (a.len() * b.len()) as f64
        }
    }
}

/// Compute the silhouette score for a given clustering.
///
/// Returns the mean silhouette coefficient across all points.
pub fn silhouette_score(data: &[Vec<Ternary>], assignments: &[usize]) -> f64 {
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }

    let k = *assignments.iter().max().unwrap_or(&0) + 1;
    if k <= 1 {
        return 0.0;
    }

    let mut total_score = 0.0f64;

    for i in 0..n {
        let ci = assignments[i];

        // Compute average intra-cluster distance (a)
        let same_cluster: Vec<usize> = (0..n).filter(|&j| j != i && assignments[j] == ci).collect();
        let a = if same_cluster.is_empty() {
            0.0
        } else {
            let sum: f64 = same_cluster.iter().map(|&j| hamming_distance(&data[i], &data[j]) as f64).sum();
            sum / same_cluster.len() as f64
        };

        // Compute minimum average inter-cluster distance (b)
        let mut b = f64::MAX;
        for c in 0..k {
            if c == ci {
                continue;
            }
            let other: Vec<usize> = (0..n).filter(|&j| assignments[j] == c).collect();
            if other.is_empty() {
                continue;
            }
            let avg: f64 = other.iter().map(|&j| hamming_distance(&data[i], &data[j]) as f64).sum::<f64>()
                / other.len() as f64;
            if avg < b {
                b = avg;
            }
        }

        if b == f64::MAX {
            b = 0.0;
        }

        let s = if a.max(b) == 0.0 { 0.0 } else { (b - a) / a.max(b) };
        total_score += s;
    }

    total_score / n as f64
}

/// Compute the Davies-Bouldin index for a given clustering (lower is better).
pub fn davies_bouldin_index(data: &[Vec<Ternary>], assignments: &[usize], centroids: &[Vec<Ternary>]) -> f64 {
    let k = centroids.len();
    if k <= 1 {
        return 0.0;
    }

    // Compute average distance to centroid for each cluster
    let mut avg_dist: Vec<f64> = vec![0.0; k];
    let mut counts: Vec<usize> = vec![0; k];

    for (i, point) in data.iter().enumerate() {
        let c = assignments[i];
        avg_dist[c] += (hamming_distance(point, &centroids[c])) as f64;
        counts[c] += 1;
    }

    for c in 0..k {
        if counts[c] > 0 {
            avg_dist[c] /= counts[c] as f64;
        }
    }

    let mut db_sum = 0.0;
    for i in 0..k {
        if counts[i] == 0 {
            continue;
        }
        let mut max_ratio = 0.0;
        for j in 0..k {
            if i == j || counts[j] == 0 {
                continue;
            }
            let centroid_dist = hamming_distance(&centroids[i], &centroids[j]) as f64;
            if centroid_dist == 0.0 {
                continue;
            }
            let ratio = (avg_dist[i] + avg_dist[j]) / centroid_dist;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        db_sum += max_ratio;
    }

    let non_empty = counts.iter().filter(|&&c| c > 0).count();
    if non_empty == 0 {
        return 0.0;
    }
    db_sum / non_empty as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: i8) -> Ternary {
        Ternary::from_i8(v)
    }

    fn tv(vals: &[i8]) -> Vec<Ternary> {
        vals.iter().map(|&v| t(v)).collect()
    }

    #[test]
    fn test_ternary_to_i8() {
        assert_eq!(Ternary::Neg.to_i8(), -1);
        assert_eq!(Ternary::Zero.to_i8(), 0);
        assert_eq!(Ternary::Pos.to_i8(), 1);
    }

    #[test]
    fn test_ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Ternary::Neg);
        assert_eq!(Ternary::from_i8(0), Ternary::Zero);
        assert_eq!(Ternary::from_i8(1), Ternary::Pos);
        assert_eq!(Ternary::from_i8(5), Ternary::Pos);
        assert_eq!(Ternary::from_i8(-5), Ternary::Neg);
    }

    #[test]
    fn test_sq_distance_identical() {
        let a = tv(&[1, 0, -1, 1]);
        assert_eq!(ternary_sq_distance(&a, &a), 0);
    }

    #[test]
    fn test_sq_distance_different() {
        let a = tv(&[1, 0, -1]);
        let b = tv(&[-1, 0, 1]);
        // (1-(-1))^2 + (0-0)^2 + (-1-1)^2 = 4 + 0 + 4 = 8
        assert_eq!(ternary_sq_distance(&a, &b), 8);
    }

    #[test]
    fn test_hamming_distance() {
        let a = tv(&[1, 0, -1]);
        let b = tv(&[1, 0, 0]);
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn test_hamming_distance_all_same() {
        let a = tv(&[1, -1, 0, 1]);
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_compute_centroid() {
        let points = vec![
            tv(&[1, 1]),
            tv(&[1, 1]),
            tv(&[-1, -1]),
        ];
        let c = compute_centroid(&points);
        // mean: (1/3, 1/3) => rounds to (0, 0) actually...
        // mean[0] = (1+1-1)/3 = 1/3, mean[1] = (1+1-1)/3 = 1/3
        // 1/3 < 0.5, so Zero for both
        assert_eq!(c, tv(&[0, 0]));
    }

    #[test]
    fn test_compute_centroid_empty() {
        let c = compute_centroid(&[]);
        assert!(c.is_empty());
    }

    #[test]
    fn test_nearest_centroid() {
        let point = tv(&[1, 1]);
        let centroids = vec![
            tv(&[-1, -1]),
            tv(&[1, 1]),
            tv(&[0, 0]),
        ];
        assert_eq!(nearest_centroid(&point, &centroids), 1);
    }

    #[test]
    fn test_kmeans_simple() {
        let data = vec![
            tv(&[1, 1, 1]),
            tv(&[1, 1, 0]),
            tv(&[-1, -1, -1]),
            tv(&[-1, -1, 0]),
        ];
        let (assignments, centroids) = kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(centroids.len(), 2);
        // First two should be in same cluster, last two in same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_kmeans_empty() {
        let (a, c) = kmeans(&[], 2, 10);
        assert!(a.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn test_kmeans_k_zero() {
        let (a, c) = kmeans(&[tv(&[1])], 0, 10);
        assert!(a.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn test_dbscan_two_clusters() {
        let data = vec![
            tv(&[1, 1, 1]),
            tv(&[1, 1, 0]),
            tv(&[-1, -1, -1]),
            tv(&[-1, -1, 0]),
        ];
        let labels = dbscan(&data, 1, 2);
        assert_eq!(labels.len(), 4);
        // First two should be in one cluster, last two in another
        assert!(labels[0].is_some());
        assert!(labels[1].is_some());
        assert!(labels[2].is_some());
        assert!(labels[3].is_some());
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_dbscan_noise() {
        let data = vec![
            tv(&[1, 1, 1]),
            tv(&[-1, -1, -1]),
            tv(&[0, 0, 0]), // outlier
        ];
        let labels = dbscan(&data, 0, 2);
        // With eps=0, each point is only its own neighbor, so no clusters formed
        assert!(labels.iter().all(|l| l.is_none()));
    }

    #[test]
    fn test_dbscan_empty() {
        let labels = dbscan(&[], 1, 2);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_hierarchical_single() {
        let data = vec![
            tv(&[1, 1]),
            tv(&[1, 0]),
            tv(&[-1, -1]),
        ];
        let merges = hierarchical(&data, Linkage::Single);
        assert_eq!(merges.len(), 2); // n-1 merges
        // First merge should be between closest pair (0, 1) with distance 1
        assert_eq!(merges[0].2, 1.0);
    }

    #[test]
    fn test_hierarchical_complete() {
        let data = vec![
            tv(&[1, 1]),
            tv(&[1, 0]),
            tv(&[-1, -1]),
        ];
        let merges = hierarchical(&data, Linkage::Complete);
        assert_eq!(merges.len(), 2);
    }

    #[test]
    fn test_hierarchical_average() {
        let data = vec![
            tv(&[1, 1]),
            tv(&[1, 0]),
            tv(&[-1, -1]),
        ];
        let merges = hierarchical(&data, Linkage::Average);
        assert_eq!(merges.len(), 2);
    }

    #[test]
    fn test_hierarchical_single_point() {
        let data = vec![tv(&[1, 1])];
        let merges = hierarchical(&data, Linkage::Single);
        assert!(merges.is_empty());
    }

    #[test]
    fn test_silhouette_perfect_clusters() {
        // Two well-separated clusters
        let data = vec![
            tv(&[1, 1, 1]),
            tv(&[1, 1, 0]),
            tv(&[-1, -1, -1]),
            tv(&[-1, -1, 0]),
        ];
        let assignments = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &assignments);
        assert!(score > 0.5, "Silhouette should be high for well-separated clusters, got {}", score);
    }

    #[test]
    fn test_silhouette_single_cluster() {
        let data = vec![tv(&[1, 0]), tv(&[-1, 0])];
        let assignments = vec![0, 0];
        let score = silhouette_score(&data, &assignments);
        assert_eq!(score, 0.0); // Only one cluster => silhouette = 0
    }

    #[test]
    fn test_davies_bouldin() {
        let data = vec![
            tv(&[1, 1]),
            tv(&[1, 0]),
            tv(&[-1, -1]),
            tv(&[-1, 0]),
        ];
        let assignments = vec![0, 0, 1, 1];
        let centroids = vec![tv(&[1, 1]), tv(&[-1, -1])];
        let db = davies_bouldin_index(&data, &assignments, &centroids);
        assert!(db >= 0.0);
    }

    #[test]
    fn test_davies_bouldin_single_cluster() {
        let data = vec![tv(&[1, 0])];
        let assignments = vec![0];
        let centroids = vec![tv(&[1, 0])];
        let db = davies_bouldin_index(&data, &assignments, &centroids);
        assert_eq!(db, 0.0);
    }
}

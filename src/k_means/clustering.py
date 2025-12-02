from __future__ import division, print_function
import argparse
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import from logreg
sys.path.append(str(Path(__file__).parent.parent))
from logreg.util import get_keypoint_names, extract_keypoint_coordinates

def load_keypoints_data(keypoints_dir, labels_df):
    """
    Load keypoints data from JSON files and match with labels.
    Aggregates features across all frames in a video.

    Args:
        keypoints_dir: Path to directory containing keypoints subdirectories
        labels_df: DataFrame with labels

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        sample_ids: List of sample identifiers
    """
    X_list = []
    y_list = []
    sample_ids = []

    for idx, row in labels_df.iterrows():
        match_id = row['match_id']
        pt = row['Pt']
        serve_result = row['Serve Result']

        # Determine serve number (1 or 2) based on whether 1st or 2nd serve
        if pd.notna(row['1st']):
            serve_num = 1
        elif pd.notna(row['2nd']):
            serve_num = 2
        else:
            continue  # Skip if no serve data

        # Construct directory name
        dir_name = f"{match_id}_{pt}_{serve_num}"
        serve_dir = os.path.join(keypoints_dir, dir_name)

        if not os.path.exists(serve_dir):
            print(f"Warning: Directory not found: {serve_dir}")
            continue

        # Load all keypoints JSON files in this directory
        json_files = sorted([f for f in os.listdir(serve_dir) if f.endswith('_keypoints.json')])

        if len(json_files) == 0:
            print(f"Warning: No keypoints files in {serve_dir}")
            continue

        # Extract features from all frames
        frame_features = []
        for json_file in json_files:
            json_path = os.path.join(serve_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract pose keypoints from first person detected
            if len(data['people']) > 0:
                keypoints = data['people'][0]['pose_keypoints_2d']
                frame_features.append(keypoints)

        if len(frame_features) == 0:
            continue

        # Aggregate features across all frames in the video
        frame_features = np.array(frame_features)  # shape: (n_frames, 75)
        mean_features = np.mean(frame_features, axis=0)
        std_features = np.std(frame_features, axis=0)
        max_features = np.max(frame_features, axis=0)
        min_features = np.min(frame_features, axis=0)

        # Concatenate all statistics into a single feature vector
        combined_features = np.concatenate([mean_features, std_features, max_features, min_features])

        X_list.append(combined_features)
        y_list.append(serve_result)
        sample_ids.append(dir_name)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, sample_ids


def init_centroids(num_clusters, data):
    """
    Initialize centroids by randomly selecting data points.
    
    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    data : nparray
        (n_samples, n_features) data represented as an nparray
    
    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids 
    """
    centroids_init = np.zeros((num_clusters, data.shape[1]), dtype=int)

    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    random_indices = np.random.choice(n_samples, size=num_clusters, replace=False)
    centroids_init = data[random_indices].copy()
    
    return centroids_init


def update_centroids(centroids, data, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times.
    
    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    data : nparray
        (n_samples, n_features) data reprsented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update
    
    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """
    n_samples = data.shape[0]
    n_features = data.shape[1]
    num_clusters = centroids.shape[0]
    new_centroids = centroids.copy()
    
    for i in range(max_iter):
        dist_vec = np.zeros((n_samples, num_clusters), dtype=int)
        # intialising distance vector to keep track of distance to every centroid
        for centroid in range(num_clusters):
            dist_vec[:, centroid] = np.sum((data - new_centroids[centroid])**2, axis=1)
        
        closest_centroid = np.argmin(dist_vec, axis=1) # creates array of closest for all coordinates
        
        # iterates through clusters and takes mean value of each cluster to update centroid
        for centroid in range(num_clusters):
            if np.any(closest_centroid == centroid):
                new_centroids[centroid] = data[closest_centroid == centroid].mean(axis=0)
        
        if ((i + 1) % print_every) == 0:
            print("Still updating centroids at iteration: ", i + 1)
    
    return new_centroids


def assign_to_centroids(data, centroids):
    """
    Assign each data point to the closest centroid.
    
    Parameters
    ----------
    data : nparray
        (n_samples, n_features) data matrix
    centroids : nparray
        Centroids of shape (num_clusters, n_features)
    
    Returns
    -------
    assignments : nparray
        Cluster assignments for each data point, shape (n_samples,)
    """
    n_samples = data.shape[0]
    num_clusters = centroids.shape[0]
    
    dist_vec = np.zeros((n_samples, num_clusters))
    
    for centroid in range(num_clusters):
        dist_vec[:, centroid] = np.sum((data - centroids[centroid])**2, axis=1)
    
    assignments = np.argmin(dist_vec, axis=1)
    
    return assignments


def main(args):
    """
    Main function to run k-means clustering on pose estimation data.
    """
    # Setup paths
    project_dir = Path(__file__).parent.parent.parent
    keypoints_dir = project_dir / 'data' / 'keypoints'
    labels_path = project_dir / 'data' / 'mcp_data.csv'
    output_dir = project_dir / 'src' / 'k_means'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print("Loading labels from CSV...")
    df = pd.read_csv(labels_path)
    print(f"Total samples in CSV: {len(df)}")
    
    # Use training set for clustering (or all data if specified)
    if args.use_test_set:
        df_cluster = df[df['Test Set'] == 'Test'].copy()
    else:
        df_cluster = df[df['Test Set'] != 'Test'].copy()
    
    print(f"Samples for clustering: {len(df_cluster)}")
    
    # Load data
    print("\nLoading full pose keypoints data...")
    X, y, sample_ids = load_keypoints_data(str(keypoints_dir), df_cluster)
    feature_name = "pose"
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    
    if len(X) == 0:
        print("Error: No data loaded!")
        return
    
    # Normalize features (important for k-means)
    print("\nNormalizing features...")
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    X_norm = (X - mean) / std
    
    # Initialize centroids
    print(f"\n[INFO] Initializing {args.num_clusters} centroids...")
    centroids_init = init_centroids(args.num_clusters, X_norm)
    
    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, X_norm, args.max_iter, args.print_every)
    
    # Assign data points to clusters
    print("\nAssigning data points to clusters...")
    assignments = assign_to_centroids(X_norm, centroids)
    
    # Print cluster statistics
    print("\n" + "="*50)
    print("Cluster Statistics")
    print("="*50)
    for k in range(args.num_clusters):
        mask = assignments == k
        count = np.sum(mask)
        print(f"Cluster {k}: {count} samples ({100*count/len(X):.1f}%)")
    
    # Save results
    results_path = output_dir / f'clustering_results_{feature_name}_k{args.num_clusters}.txt'
    with open(results_path, 'w') as f:
        f.write(f"K-Means Clustering Results\n")
        f.write(f"Feature type: {feature_name}\n")
        f.write(f"Number of clusters: {args.num_clusters}\n")
        f.write(f"Number of samples: {len(X)}\n")
        f.write(f"Number of features: {X.shape[1]}\n\n")
        
        f.write("Cluster Assignments:\n")
        for i, (sample_id, cluster) in enumerate(zip(sample_ids, assignments)):
            f.write(f"{sample_id}: Cluster {cluster}\n")
        
        f.write("\n\nCentroids:\n")
        for k in range(args.num_clusters):
            f.write(f"\nCluster {k} centroid (first 10 features):\n")
            f.write(f"{centroids[k][:10]}\n")
    
    print(f"\nResults saved to {results_path}")
    
    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='K-means clustering on pose estimation data')
    parser.add_argument('--num_clusters', type=int, default=5,
                        help='Number of centroids/clusters')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    parser.add_argument('--use_test_set', action='store_true',
                        help='Use test set instead of training set for clustering')
    args = parser.parse_args()
    main(args)

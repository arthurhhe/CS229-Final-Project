"""Utility functions for tennis serve classification."""

import numpy as np
import json
import os
from typing import Tuple, List, Dict


def load_single_serve_keypoints(serve_dir: str, aggregate: bool = True) -> np.ndarray:
    """
    Load keypoints from all frames in a serve directory.
    
    Args:
        serve_dir: Path to directory containing keypoints JSON files
        aggregate: If True, aggregate across frames; otherwise return all frames
    
    Returns:
        Feature vector (aggregated) or matrix (all frames)
    """
    json_files = sorted([f for f in os.listdir(serve_dir) 
                        if f.endswith('_keypoints.json')])
    
    if len(json_files) == 0:
        raise ValueError(f"No keypoints files found in {serve_dir}")
    
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
        raise ValueError(f"No person detected in any frame in {serve_dir}")
    
    frame_features = np.array(frame_features)
    
    if aggregate:
        # Compute statistics across frames
        mean_features = np.mean(frame_features, axis=0)
        std_features = np.std(frame_features, axis=0)
        max_features = np.max(frame_features, axis=0)
        min_features = np.min(frame_features, axis=0)
        
        # Concatenate all statistics
        aggregated = np.concatenate([mean_features, std_features, 
                                     max_features, min_features])
        return aggregated
    else:
        return frame_features


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: Array of class labels
        num_classes: Number of classes
    
    Returns:
        Array of class weights
    """
    class_counts = np.bincount(y, minlength=num_classes)
    total_samples = len(y)
    
    # Inverse frequency weighting
    weights = total_samples / (num_classes * class_counts + 1e-6)
    
    return weights


def get_keypoint_names() -> List[str]:
    """
    Get names of the 25 COCO body keypoints.
    
    Returns:
        List of keypoint names
    """
    return [
        'Nose',           # 0
        'Neck',           # 1
        'RShoulder',      # 2
        'RElbow',         # 3
        'RWrist',         # 4
        'LShoulder',      # 5
        'LElbow',         # 6
        'LWrist',         # 7
        'MidHip',         # 8
        'RHip',           # 9
        'RKnee',          # 10
        'RAnkle',         # 11
        'LHip',           # 12
        'LKnee',          # 13
        'LAnkle',         # 14
        'REye',           # 15
        'LEye',           # 16
        'REar',           # 17
        'LEar',           # 18
        'LBigToe',        # 19
        'LSmallToe',      # 20
        'LHeel',          # 21
        'RBigToe',        # 22
        'RSmallToe',      # 23
        'RHeel'           # 24
    ]


def extract_keypoint_coordinates(keypoints_2d: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract x, y coordinates and confidence scores from keypoints array.
    
    Args:
        keypoints_2d: Flat array of [x1, y1, c1, x2, y2, c2, ...]
    
    Returns:
        Tuple of (x_coords, y_coords, confidences)
    """
    keypoints_array = np.array(keypoints_2d)
    keypoints_reshaped = keypoints_array.reshape(-1, 3)
    
    x_coords = keypoints_reshaped[:, 0]
    y_coords = keypoints_reshaped[:, 1]
    confidences = keypoints_reshaped[:, 2]
    
    return x_coords, y_coords, confidences


def compute_velocities(frame_features: np.ndarray) -> np.ndarray:
    """
    Compute velocities between consecutive frames.
    
    Args:
        frame_features: Array of shape (n_frames, n_keypoint_values)
    
    Returns:
        Velocity features (mean, std, max of velocities across frames)
    """
    if len(frame_features) < 2:
        # Not enough frames to compute velocity
        return np.zeros(frame_features.shape[1] * 3)
    
    velocities = np.diff(frame_features, axis=0)
    
    mean_vel = np.mean(velocities, axis=0)
    std_vel = np.std(velocities, axis=0)
    max_vel = np.max(np.abs(velocities), axis=0)
    
    return np.concatenate([mean_vel, std_vel, max_vel])


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: List[str]) -> None:
    """
    Print a classification report with precision, recall, and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    from sklearn.metrics import classification_report
    
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   zero_division=0)
    print(report)


def save_model_weights(theta: np.ndarray, filepath: str) -> None:
    """
    Save model weights to a file.
    
    Args:
        theta: Weight matrix
        filepath: Path to save the weights
    """
    np.save(filepath, theta)
    print(f"Model weights saved to {filepath}")


def load_model_weights(filepath: str) -> np.ndarray:
    """
    Load model weights from a file.
    
    Args:
        filepath: Path to load the weights from
    
    Returns:
        Weight matrix
    """
    theta = np.load(filepath)
    print(f"Model weights loaded from {filepath}")
    return theta


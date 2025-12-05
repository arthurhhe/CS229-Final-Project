import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def load_keypoints_data(keypoints_dir, labels_df):
    """
    Load keypoints data from JSON files and match with labels.

    Args:
        keypoints_dir: Path to directory containing keypoints subdirectories
        labels_df: DataFrame with labels (already filtered for training/validation)

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

        # Aggregate features across frames (use statistics like mean, std, max, min)
        frame_features = np.array(frame_features)  # shape: (n_frames, 75)

        # Compute statistics across frames
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

def map_label_to_direction(label):
    """
    Map labels to 3-class directions (4, 5, 6).
    Handles both string and numeric labels.
    
    Args:
        label: String or numeric label like '4', '4d', 4, '5', '6', '6n', etc.
    
    Returns:
        Mapped label: '4', '5', or '6'
    """
    # Convert to string if not already
    label_str = str(label)
    
    if label_str.startswith('4'):
        return '4'
    elif label_str.startswith('5'):
        return '5'
    elif label_str.startswith('6'):
        return '6'
    else:
        raise ValueError(f"Unknown label: {label} (type: {type(label)})")

def plot_confusion_matrix(y_true, y_pred, class_names, output_path, title="Confusion Matrix"):
    """
    Plot and save a confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        output_path: Path to save the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def analyze_feature_importance(X_train, y_train, X_dev, y_dev, X_test, y_test, 
                                feature_names, output_dir, class_names, seed):
    """
    Train a decision tree classifier, analyze feature importance, and save predictions.

    Args:
        X_train: Training feature matrix (n_train, n_features)
        y_train: Training labels (n_train,)
        X_dev: Dev feature matrix (n_dev, n_features)
        y_dev: Dev labels (n_dev,)
        X_test: Test feature matrix (n_test, n_features)
        y_test: Test labels (n_test,)
        feature_names: List of feature names corresponding to columns in X
        output_dir: Directory to save outputs (e.g., predictions, confusion matrix)
        class_names: List of class names
    """
    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate on training set
    print("\n" + "="*50)
    print("Training Set Results")
    print("="*50)
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print("\nClassification Report (Training):")
    print(classification_report(y_train, y_train_pred, target_names=class_names))
    
    # Per-class accuracy on training set
    print("\nPer-class accuracy (Training):")
    for i, class_name in enumerate(class_names):
        mask = y_train == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_train_pred[mask] == y_train[mask])
            print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")

    # Evaluate on dev set
    print("\n" + "="*50)
    print("Dev/Eval Set Results")
    print("="*50)
    y_dev_pred = clf.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"Dev Accuracy: {dev_accuracy:.4f}")
    print("\nClassification Report (Dev):")
    print(classification_report(y_dev, y_dev_pred, target_names=class_names))
    
    # Per-class accuracy on dev set
    print("\nPer-class accuracy (Dev):")
    for i, class_name in enumerate(class_names):
        mask = y_dev == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_dev_pred[mask] == y_dev[mask])
            print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")

    # Evaluate on test set
    print("\n" + "="*50)
    print("Test Set Results")
    print("="*50)
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    # Extract feature importance
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Feature Importance:")
    print(importance_df.head(10))

    # Save feature importance to a CSV file
    feature_importance_path = output_dir / f'feature_importance_seed_{seed}.csv'
    importance_df.to_csv(feature_importance_path, index=False)
    print(f"\nFeature importance saved to {feature_importance_path}")

    # Save test predictions
    idx_to_label = {i: label for i, label in enumerate(class_names)}
    test_pred_path = output_dir / f'test_predictions_seed_{seed}.txt'
    with open(test_pred_path, 'w') as f:
        f.write("Sample,True_Label,Predicted_Label,Confidence\n")
        test_probs = clf.predict_proba(X_test)
        for i in range(len(y_test)):
            true_label = idx_to_label[y_test[i]]
            pred_label = idx_to_label[y_test_pred[i]]
            confidence = test_probs[i, y_test_pred[i]]
            f.write(f"{i},{true_label},{pred_label},{confidence:.4f}\n")
    print(f"Test predictions saved to {test_pred_path}")

    # Plot confusion matrix for test set
    cm_test_path = output_dir / f'confusion_matrix_test_seed_{seed}.png'
    plot_confusion_matrix(y_test, y_test_pred, class_names, cm_test_path, 
                         title="Confusion Matrix - Decision Tree")

    # Per-class accuracy for test set
    print("\nPer-class accuracy (Test):")
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_test_pred[mask] == y_test[mask])
            print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # Save accuracy summary
    summary_path = output_dir / f'decisiontree_accuracy_summary_seed_{seed}.txt'
    with open(summary_path, 'w') as f:
        f.write("Decision Tree Model Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Data Split: Train=70, Dev=15, Test=15\n\n")
        f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Dev Accuracy: {dev_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write("Per-class Test Accuracy:\n")
        for i, class_name in enumerate(class_names):
            mask = y_test == i
            if np.sum(mask) > 0:
                class_acc = np.mean(y_test_pred[mask] == y_test[mask])
                f.write(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)\n")
    print(f"\nAccuracy summary saved to {summary_path}")

def main():
    """
    Main function to load data, train a decision tree classifier, and analyze feature importance.
    Randomly splits 100 examples into train (70), dev (15), and test (15).
    Maps labels to 3 classes (4, 5, 6).
    """
    # Set random seed for reproducibility
    seed = 17
    np.random.seed(seed)

    # Paths
    project_dir = Path(__file__).parent.parent.parent
    keypoints_dir = project_dir / 'data' / 'keypoints'
    labels_path = project_dir / 'data' / 'mcp_data.csv'
    output_dir = project_dir / 'src' / 'decisiontree' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define class names (only 3 classes: 4, 5, 6)
    class_names = ['4', '5', '6']

    # Load labels
    print("Loading labels from CSV...")
    df = pd.read_csv(labels_path)
    print(f"Total samples in CSV: {len(df)}")

    # Load all keypoints data first
    print("\nLoading all keypoints data...")
    X_all, y_all_raw, all_ids = load_keypoints_data(str(keypoints_dir), df)
    print(f"Loaded {len(X_all)} total samples with {X_all.shape[1]} features each")

    if len(X_all) == 0:
        print("Error: No data loaded!")
        return

    # Map labels to 3 classes (4, 5, 6)
    print("\nMapping labels to 3 classes (4, 5, 6)...")
    direction_to_idx = {d: i for i, d in enumerate(class_names)}
    
    # Convert labels to strings if they're not already (handles numeric labels)
    y_all_raw_str = [str(label) for label in y_all_raw]
    y_all_mapped = [map_label_to_direction(label) for label in y_all_raw_str]
    y_all = np.array([direction_to_idx[label] for label in y_all_mapped])
    
    print(f"Label distribution (all data):")
    for i, cls in enumerate(class_names):
        count = np.sum(y_all == i)
        print(f"  {cls}: {count} samples")

    # Randomly split into train (70), dev (15), test (15)
    n_total = len(X_all)
    n_train = 70
    n_dev = 15
    n_test = 15
    
    if n_total != n_train + n_dev + n_test:
        print(f"Warning: Total samples ({n_total}) doesn't match expected split (70+15+15=100)")
        print(f"Adjusting split to match available data...")
        n_test = n_total - n_train - n_dev
    
    # Shuffle indices
    indices = np.random.permutation(n_total)
    
    train_indices = indices[:n_train]
    dev_indices = indices[n_train:n_train + n_dev]
    test_indices = indices[n_train + n_dev:n_train + n_dev + n_test]
    
    # Split data
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    
    X_dev = X_all[dev_indices]
    y_dev = y_all[dev_indices]
    
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Dev/Eval: {len(X_dev)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Normalize features (use training data statistics only)
    print("\nNormalizing features...")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_train_norm = (X_train - mean) / std
    X_dev_norm = (X_dev - mean) / std
    X_test_norm = (X_test - mean) / std

    # Generate feature names (for simplicity, use indices)
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    # Analyze feature importance and save outputs
    analyze_feature_importance(X_train_norm, y_train, X_dev_norm, y_dev, 
                              X_test_norm, y_test, feature_names, output_dir, class_names, seed)

if __name__ == '__main__':
    main()
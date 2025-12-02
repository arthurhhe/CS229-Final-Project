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

def extract_base_number(label_str):
    """
    Extract the base number from a label string.
    Examples: "4d" -> "4", "6n" -> "6", "5" -> "5"
    """
    if len(label_str) > 0 and label_str[0].isdigit():
        return label_str[0]
    return label_str

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

def analyze_feature_importance(X, y, feature_names, output_dir, class_names):
    """
    Train a decision tree classifier, analyze feature importance, and save predictions.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        feature_names: List of feature names corresponding to columns in X
        output_dir: Directory to save outputs (e.g., predictions, confusion matrix)
        class_names: List of class names
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Extract feature importance
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    # Save feature importance to a CSV file
    feature_importance_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(feature_importance_path, index=False)
    print(f"Feature importance saved to {feature_importance_path}")

    # Save test predictions
    idx_to_label = {i: label for i, label in enumerate(class_names)}
    test_pred_path = output_dir / 'test_predictions.txt'
    with open(test_pred_path, 'w') as f:
        f.write("Sample,True_Label,Predicted_Label,True_Base,Predicted_Base\n")
        for i in range(len(y_test)):
            true_label = idx_to_label[y_test[i]]
            pred_label = idx_to_label[y_pred[i]]
            true_base = extract_base_number(true_label)
            pred_base = extract_base_number(pred_label)
            f.write(f"{i},{true_label},{pred_label},{true_base},{pred_base}\n")
    print(f"Test predictions saved to {test_pred_path}")

    # Plot confusion matrix for exact labels
    cm_exact_path = output_dir / 'confusion_matrix_exact.png'
    plot_confusion_matrix(y_test, y_pred, class_names, cm_exact_path, title="Confusion Matrix (Exact Labels)")

    # Compute base number predictions
    y_test_base = np.array([extract_base_number(idx_to_label[label]) for label in y_test])
    y_pred_base = np.array([extract_base_number(idx_to_label[label]) for label in y_pred])
    base_class_names = sorted(set(y_test_base))  # Unique base numbers
    base_label_to_idx = {label: i for i, label in enumerate(base_class_names)}
    y_test_base_idx = np.array([base_label_to_idx[label] for label in y_test_base])
    y_pred_base_idx = np.array([base_label_to_idx[label] for label in y_pred_base])

    # Plot confusion matrix for base labels
    cm_base_path = output_dir / 'confusion_matrix_base.png'
    plot_confusion_matrix(y_test_base_idx, y_pred_base_idx, base_class_names, cm_base_path, title="Confusion Matrix (Base Labels)")

    # Per-class accuracy for base labels
    print("\nPer-class accuracy (Base Labels):")
    for i, base_label in enumerate(base_class_names):
        mask = y_test_base_idx == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_base_idx[mask] == y_test_base_idx[mask])
            print(f"  {base_label}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    total_accuracy = np.mean(y_pred_base_idx == y_test_base_idx)
    print(f"\nTotal Accuracy (Base Labels): {total_accuracy:.4f}")

def main():
    """
    Main function to load data, train a decision tree classifier, and analyze feature importance.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Paths
    project_dir = Path(__file__).parent.parent.parent
    keypoints_dir = project_dir / 'data' / 'keypoints'
    labels_path = project_dir / 'data' / 'mcp_data.csv'
    output_dir = project_dir / 'src' / 'decisiontree' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define class names
    class_names = ['4', '4d', '4n', '4w', '4x', '5', '6', '6d', '6n']

    # Load labels
    print("Loading labels from CSV...")
    df = pd.read_csv(labels_path)
    print(f"Total samples in CSV: {len(df)}")

    # Split based on "Test Set" column
    df_test = df[df['Test Set'] == 'Test'].copy()
    df_train = df[df['Test Set'] != 'Test'].copy()

    print(f"Training samples in CSV: {len(df_train)}")
    print(f"Test samples in CSV: {len(df_test)}")

    # Load keypoints data
    print("\nLoading keypoints data for training set...")
    X_train, y_train, train_ids = load_keypoints_data(str(keypoints_dir), df_train)
    print(f"Loaded {len(X_train)} training samples with {X_train.shape[1]} features each")

    print("\nLoading keypoints data for test set...")
    X_test, y_test, test_ids = load_keypoints_data(str(keypoints_dir), df_test)
    print(f"Loaded {len(X_test)} test samples")

    # Combine labels from training and test sets
    all_labels = np.concatenate([y_train, y_test])

    # Fit the LabelEncoder on all labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Transform labels
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Combine training and test data for feature importance analysis
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])

    # Generate feature names (for simplicity, use indices)
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # Analyze feature importance and save outputs
    analyze_feature_importance(X, y, feature_names, output_dir, class_names)

if __name__ == '__main__':
    main()
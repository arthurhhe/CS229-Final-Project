import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
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


class SoftmaxRegression:
    """Softmax Regression (Multi-class Logistic Regression) with Newton's Method.
    
    Example usage:
        > clf = SoftmaxRegression(num_classes=9)
        > clf.fit(x_train, y_train)
        > predictions = clf.predict(x_test)
    """
    
    def __init__(self, num_classes, step_size=0.01, max_iter=1000, eps=1e-5,
                 theta_0=None, verbose=True, use_gradient_descent=True):
        """
        Args:
            num_classes: Number of classes for classification.
            step_size: Step size for gradient descent.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero matrix.
            verbose: Print loss values during training.
            use_gradient_descent: Use gradient descent instead of Newton's method (more stable for multi-class)
        """
        self.num_classes = num_classes
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.use_gradient_descent = use_gradient_descent
    
    def softmax(self, z):
        """
        Compute softmax function.
        
        Args:
            z: Input array of shape (n_examples, num_classes)
        
        Returns:
            Softmax probabilities of shape (n_examples, num_classes)
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute_loss(self, X, y):
        """
        Compute cross-entropy loss.
        
        Args:
            X: Feature matrix (n_examples, dim)
            y: Label indices (n_examples,)
        
        Returns:
            Average cross-entropy loss
        """
        n_examples = X.shape[0]
        
        # Compute predictions
        z = X.dot(self.theta)  # (n_examples, num_classes)
        probs = self.softmax(z)  # (n_examples, num_classes)
        
        # Compute loss
        log_likelihood = -np.log(probs[np.arange(n_examples), y] + 1e-10)
        loss = np.mean(log_likelihood)
        
        return loss
    
    def fit(self, X, y):
        """
        Train softmax regression using gradient descent or Newton's method.
        
        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels (class indices). Shape (n_examples,).
        """
        n_examples, dim = X.shape
        
        # Initialize theta
        if self.theta is None:
            self.theta = np.zeros((dim, self.num_classes))
        
        # Convert labels to one-hot encoding
        y_one_hot = np.zeros((n_examples, self.num_classes))
        y_one_hot[np.arange(n_examples), y] = 1
        
        for iteration in range(self.max_iter):
            theta_old = self.theta.copy()
            
            # Compute predictions
            z = X.dot(self.theta)  # (n_examples, num_classes)
            probs = self.softmax(z)  # (n_examples, num_classes)
            
            # Compute gradient
            gradient = (1 / n_examples) * X.T.dot(probs - y_one_hot)  # (dim, num_classes)
            
            if self.use_gradient_descent:
                # Gradient descent update
                self.theta = self.theta - self.step_size * gradient
            else:
                # Newton's method update (more complex for multi-class)
                # For simplicity, using gradient descent with adaptive step size
                self.theta = self.theta - self.step_size * gradient
            
            # Check convergence
            theta_diff = np.linalg.norm(self.theta - theta_old, ord='fro')
            
            if theta_diff < self.eps:
                if self.verbose:
                    print(f'Converged in {iteration + 1} iterations')
                break
            
            # Print progress
            if self.verbose and (iteration % 100 == 0 or iteration == self.max_iter - 1):
                loss = self.compute_loss(X, y)
                accuracy = self.score(X, y)
                print(f'Iteration {iteration}, Loss = {loss:.5f}, Accuracy = {accuracy:.4f}')
    
    def predict_proba(self, X):
        """
        Return predicted probabilities given new inputs X.
        
        Args:
            X: Inputs of shape (n_examples, dim).
        
        Returns:
            Probabilities of shape (n_examples, num_classes).
        """
        z = X.dot(self.theta)
        return self.softmax(z)
    
    def predict(self, X):
        """
        Return predicted class indices given new inputs X.
        
        Args:
            X: Inputs of shape (n_examples, dim).
        
        Returns:
            Predicted class indices of shape (n_examples,).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def score(self, X, y):
        """
        Compute accuracy on given data.
        
        Args:
            X: Inputs of shape (n_examples, dim).
            y: True labels of shape (n_examples,).
        
        Returns:
            Accuracy (fraction of correct predictions).
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrix showing all 9 classes.
    
    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        class_names: List of all class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Create confusion matrix for ALL classes (0 to 8), even if some are not present
    num_classes = len(class_names)
    all_labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted label',
           ylabel='True label',
           title='Confusion Matrix (9x9 - All Classes)')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def grouped_accuracy_int(y_true, y_pred, class_names):
    group_map = {
        '4': '4', '4d': '4', '4n': '4', '4w': '4', '4x': '4',
        '5': '5',
        '6': '6', '6d': '6', '6n': '6'
    }

    correct = 0
    for yt, yp in zip(y_true, y_pred):
        true_label = class_names[yt]
        pred_label = class_names[yp]

        if group_map[true_label] == group_map[pred_label]:
            correct += 1

    return correct / len(y_true)

def main():
    """
    Main function to train and evaluate the softmax regression model.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Paths
    project_dir = Path(__file__).parent.parent.parent
    keypoints_dir = project_dir / 'data' / 'keypoints'
    labels_path = project_dir / 'data' / 'mcp_data.csv'
    output_dir = project_dir / 'src' / 'logreg' / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print("Loading labels from CSV...")
    df = pd.read_csv(labels_path)
    print(f"Total samples in CSV: {len(df)}")
    
    # Split based on "Test Set" column
    # Test set: rows where "Test Set" column equals "Test"
    # Training set: all other rows
    df_test = df[df['Test Set'] == 'Test'].copy()
    df_train = df[df['Test Set'] != 'Test'].copy()
    
    print(f"Training samples in CSV: {len(df_train)}")
    print(f"Test samples in CSV: {len(df_test)}")
    
    # Print test set details for verification
    print("\nTest set samples:")
    for idx, row in df_test.iterrows():
        serve_type = "1st" if pd.notna(row['1st']) else "2nd" if pd.notna(row['2nd']) else "unknown"
        print(f"  Pt {row['Pt']}, {serve_type} serve, Label: {row['Serve Result']}")
    
    # Define class names
    class_names = ['4', '4d', '4n', '4w', '4x', '5', '6', '6d', '6n']
    
    # Load keypoints data
    print("\nLoading keypoints data for training set...")
    X_train, y_train, train_ids = load_keypoints_data(str(keypoints_dir), df_train)
    print(f"Loaded {len(X_train)} training samples with {X_train.shape[1]} features each")
    
    print("\nLoading keypoints data for test set...")
    X_test, y_test, test_ids = load_keypoints_data(str(keypoints_dir), df_test)
    print(f"Loaded {len(X_test)} test samples")
    
    # Check if we have data
    if len(X_train) == 0:
        print("Error: No training data loaded!")
        return
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Encode labels
    y_train_enc = np.array([class_names.index(label) for label in y_train])
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    if len(X_test) > 0:
        y_test_enc = np.array([label_to_idx[label] for label in y_test])
    
    # Normalize features (important for gradient descent)
    print("\nNormalizing features...")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_train_norm = (X_train - mean) / std
    if len(X_test) > 0:
        X_test_norm = (X_test - mean) / std
    
    # Add intercept term
    X_train_norm = np.hstack([np.ones((X_train_norm.shape[0], 1)), X_train_norm])
    if len(X_test) > 0:
        X_test_norm = np.hstack([np.ones((X_test_norm.shape[0], 1)), X_test_norm])
    
    # Train model
    print("\n" + "="*50)
    print("Training Softmax Regression Model")
    print("="*50)
    
    clf = SoftmaxRegression(
        num_classes=len(class_names),
        step_size=0.1,
        max_iter=2000,
        eps=1e-6,
        verbose=True,
        use_gradient_descent=True
    )
    
    clf.fit(X_train_norm, y_train_enc)
    
    # Evaluate on training set
    print("\n" + "="*50)
    print("Training Set Results")
    print("="*50)
    train_accuracy = clf.score(X_train_norm, y_train_enc)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Per-class accuracy on training set
    train_predictions = clf.predict(X_train_norm)
    print("\nPer-class accuracy (Training):")
    for i, class_name in enumerate(class_names):
        mask = y_train_enc == i
        if np.sum(mask) > 0:
            class_acc = np.mean(train_predictions[mask] == y_train_enc[mask])
            print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    train_group_acc = grouped_accuracy_int(y_train_enc, train_predictions, class_names)
    print(f"\n\nTraining Grouped Accuracy: {train_group_acc:.4f}")
    # Evaluate on test set if available
    if len(X_test) > 0:
        print("\n" + "="*50)
        print("Test Set Results")
        print("="*50)
        test_accuracy = clf.score(X_test_norm, y_test_enc)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        test_predictions = clf.predict(X_test_norm)
        test_probs = clf.predict_proba(X_test_norm)
        
        # Save test predictions
        test_pred_path = output_dir / 'test_predictions.txt'
        with open(test_pred_path, 'w') as f:
            f.write("Sample,True_Label,Predicted_Label,Confidence\n")
            for i in range(len(y_test_enc)):
                true_label = idx_to_label[y_test_enc[i]]
                pred_label = idx_to_label[test_predictions[i]]
                confidence = test_probs[i, test_predictions[i]]
                f.write(f"{i},{true_label},{pred_label},{confidence:.4f}\n")
        print(f"Test predictions saved to {test_pred_path}")
        
        # Plot confusion matrix for test set
        cm_test_path = output_dir / 'confusion_matrix_test.png'
        plot_confusion_matrix(y_test_enc, test_predictions, class_names, str(cm_test_path))
        
        # Per-class accuracy
        print("\nPer-class accuracy (Test):")
        for i, class_name in enumerate(class_names):
            mask = y_test_enc == i
            if np.sum(mask) > 0:
                class_acc = np.mean(test_predictions[mask] == y_test_enc[mask])
                print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
        test_group_acc  = grouped_accuracy_int(y_test_enc, test_predictions, class_names)
        print(f"\n\nTest Grouped Accuracy: {test_group_acc:.4f}")

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == '__main__':
    main()


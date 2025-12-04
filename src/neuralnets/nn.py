import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import sys
from sklearn.metrics import confusion_matrix

# Add parent directory to path to import from logreg
sys.path.append(str(Path(__file__).parent.parent))
from logreg.logreg import load_keypoints_data


def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistant when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    batch_size = x.shape[0]
    number_of_classes = x.shape[1]

    softmax = np.zeros((batch_size, number_of_classes))

    for i in range(batch_size):
        max_value = np.max(x[i])  # FIX: Use np.max to get the maximum value, not np.argmax
        row = np.exp(x[i] - max_value)
        denominator = np.sum(row)
        softmax[i] = row / denominator  # Vectorized assignment is more efficient

    return softmax

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Compute the ReLU (Rectified Linear Unit) function for the input.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the ReLU results
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Compute the derivative of ReLU function.

    Args:
        x: A numpy float array (input to ReLU)

    Returns:
        A numpy float array containing the derivative (1 where x > 0, 0 otherwise)
    """
    return (x > 0).astype(float)


def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """
    W1 = np.random.normal(size=(input_size, num_hidden))
    W2 = np.random.normal(size=(num_hidden, num_output))
    b1 = np.zeros(num_hidden)
    b2 = np.zeros(num_output)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_prop(data, one_hot_labels, params, activation_func=sigmoid):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        activation_func: Activation function to use in hidden layer (sigmoid or relu)

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after activation function) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    batch_size = data.shape[0]
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # run through activation function
    hidden_activations = activation_func(data @ W1 + b1)

    # run through softmax
    output_activations = softmax(hidden_activations @ W2 + b2)

    # average loss
    J_MB = -(1 / batch_size) * np.sum(one_hot_labels * np.log(output_activations + 1e-10))
    return hidden_activations, output_activations, J_MB

def backward_prop(data, one_hot_labels, params, forward_prop_func, activation_derivative=None):
    """
    Implement the backward propagation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        activation_derivative: Function to compute derivative of activation (for sigmoid: a*(1-a), for ReLU: relu_derivative)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    batch_size = data.shape[0]
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # compute values of z_1,z_2, a_1, a_2, t
    a_1, a_2, J_MB = forward_prop_func(data, one_hot_labels, params)

    # compute dJ/dt=(t-y)
    dJ_dt = a_2 - one_hot_labels #p - e_y

    # calculate activation derivative based on activation function
    if activation_derivative is None:
        activation_deriv = a_1 * (1 - a_1) # default is sigmoid 
    else:
        activation_deriv = activation_derivative(data @ W1 + b1) # use ReLU if specified

    # compute dJ/dz_j= (t-y) * W2.T * activation_derivative
    dJ_dzj = (dJ_dt @ W2.T) * activation_deriv

    # compute dJ_dW2, dJ_db2, dJ_dW1, dJ_db1
    dJ_dW2 = (a_1.T @ dJ_dt) / batch_size
    dJ_db2 = np.sum(dJ_dt, axis=0) / batch_size
    dJ_dW1 = (data.T @ dJ_dzj) / batch_size
    dJ_db1 = np.sum(dJ_dzj, axis=0) / batch_size

    return {"W2": dJ_dW2, "b2":dJ_db2, "W1": dJ_dW1, "b1": dJ_db1}


def backward_prop_regularized(data, one_hot_labels, params, forward_prop_func, reg, activation_derivative=None):
    """
    Implement the backward propagation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)
        activation_derivative: Function to compute derivative of activation

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    batch_size = data.shape[0]
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # compute values of z_1,z_2, a_1, a_2, t
    a_1, a_2, J_MB = forward_prop_func(data, one_hot_labels, params)

    # compute dJ/dt=(t-y)
    dJ_dt = a_2 - one_hot_labels # p - e_y

     # calculate activation derivative based on activation function
    if activation_derivative is None:
        activation_deriv = a_1 * (1 - a_1) # default is sigmoid 
    else:
        activation_deriv = activation_derivative(data @ W1 + b1) # use ReLU if specified

    # compute dJ/dz_j= (t-y) * W2.T * activation_derivative
    dJ_dzj = (dJ_dt @ W2.T) * activation_deriv

    # compute dJ_dW2, dJ_db2, dJ_dW1, dJ_db1
    dJ_dW2 = (a_1.T @ dJ_dt) / batch_size + (2*reg * W2)
    dJ_db2 = np.sum(dJ_dt, axis=0) / batch_size
    dJ_dW1 = (data.T @ dJ_dzj) / batch_size + (2*reg * W1)
    dJ_db1 = np.sum(dJ_dzj, axis=0) / batch_size
    return {"W2": dJ_dW2, "b2": dJ_db2, "W1": dJ_dW1, "b1": dJ_db1}


def gradient_descent_epoch(train_data, one_hot_train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        one_hot_train_labels: A numpy array containing the one-hot embeddings of the training labels e_y.
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """
    dataset_size = train_data.shape[0]
    for i in range(0, dataset_size, batch_size):
        j = i + batch_size
        if j > dataset_size:
            j = dataset_size

        backprop_dict = backward_prop_func(train_data[i:j], one_hot_train_labels[i:j], params, forward_prop_func)

        update_W1 = backprop_dict["W1"]
        update_b1 = backprop_dict["b1"]
        update_W2 = backprop_dict["W2"]
        update_b2 = backprop_dict["b2"]

        params["W1"] -= learning_rate * update_W1
        params["b1"] -= learning_rate * update_b1
        params["W2"] -= learning_rate * update_W2
        params["b2"] -= learning_rate * update_b2
        
    return

# this fn is very different to the original 
def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=0.01, num_epochs=100, batch_size=32, reg=0.0, verbose=True,
    activation_func=sigmoid, activation_derivative=None, early_stopping=True, patience=10):

    (nexp, dim) = train_data.shape
    num_classes = train_labels.shape[1]

    params = get_initial_params_func(dim, num_hidden, num_classes)
    best_params = None
    best_dev_acc = -1
    patience_counter = 0

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        train_acc = compute_accuracy(output, train_labels)
        accuracy_train.append(train_acc)
        
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        dev_acc = compute_accuracy(output, dev_labels)
        accuracy_dev.append(dev_acc)
        
        # Early stopping: save best model and check if dev accuracy improved
        if early_stopping:
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_params = {k: v.copy() for k, v in params.items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}: Dev accuracy did not improve for {patience} epochs')
                params = best_params
                break
        
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f'Epoch {epoch}: Train Loss = {cost_train[-1]:.4f}, Train Acc = {accuracy_train[-1]:.4f}, '
                  f'Dev Loss = {cost_dev[-1]:.4f}, Dev Acc = {accuracy_dev[-1]:.4f}')

    # Use best params if early stopping was used
    if early_stopping and best_params is not None:
        params = best_params

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) == 
        np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

# if we just want to compare base rather than exact use this
def extract_base_number(label_str):
    """
    Extract the base number from a label string.
    Examples: "4d" -> "4", "6n" -> "6", "5" -> "5"
    """
    if len(label_str) > 0 and label_str[0].isdigit():
        return label_str[0]
    return label_str

def map_to_base_label(label_str):
    """
    Map labels like '4d','4n','4x' -> '4', and '6d','6n' -> '6'.
    Leaves '5' unchanged. Returns the string label.
    """
    if label_str.startswith('4'):
        return '4'
    if label_str.startswith('6'):
        return '6'
    return label_str  # e.g. "5"

def one_hot_labels(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels

# use when testing with k_means as opposed to raw data
def load_cluster_assignments(clustering_results_path, sample_ids):
    """
    Load cluster assignments from clustering results file.
    
    Args:
        clustering_results_path: Path to clustering results file
        sample_ids: List of sample IDs to match
    
    Returns:
        Dictionary mapping sample_id to cluster assignment
    """
    cluster_dict = {}
    
    if not os.path.exists(clustering_results_path):
        print(f"Warning: Clustering results file not found: {clustering_results_path}")
        return cluster_dict
    
    with open(clustering_results_path, 'r') as f:
        lines = f.readlines()
    
    in_assignments = False
    for line in lines:
        line = line.strip()
        if line == "Cluster Assignments:":
            in_assignments = True
            continue
        if in_assignments and ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                sample_id = parts[0].strip()
                cluster_str = parts[1].strip()
                if cluster_str.startswith('Cluster'):
                    cluster_num = int(cluster_str.split()[-1])
                    cluster_dict[sample_id] = cluster_num
    
    return cluster_dict


def add_cluster_features(X, sample_ids, cluster_dict, num_clusters=5):
    """
    Add cluster assignment as one-hot encoded features.
    
    Args:
        X: Feature matrix
        sample_ids: List of sample IDs
        cluster_dict: Dictionary mapping sample_id to cluster
        num_clusters: Number of clusters
    
    Returns:
        Feature matrix with cluster features added
    """
    cluster_features = np.zeros((len(sample_ids), num_clusters))
    
    for i, sample_id in enumerate(sample_ids):
        if sample_id in cluster_dict:
            cluster_features[i, cluster_dict[sample_id]] = 1.0
    
    X_with_clusters = np.hstack([X, cluster_features])
    
    return X_with_clusters


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title_suffix=""):
    """
    Plot confusion matrix showing all 9 classes.
    
    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        class_names: List of all class names
        save_path: Path to save the plot
        title_suffix: Additional text for the title
    """
    num_classes = len(class_names)
    all_labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    title = 'Confusion Matrix - Neural Network'
    if title_suffix:
        title += f' - {title_suffix}'
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted label',
           ylabel='True label',
           title=title)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
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

def get_top_features(feature_importance_path, num_features=12):
    """
    Get the indices of the top features based on importance.

    Args:
        feature_importance_path: Path to the feature importance CSV file.
        num_features: Number of top features to select.

    Returns:
        top_feature_indices: List of indices of the top features.
    """
    # Read the feature importance CSV
    importance_df = pd.read_csv(feature_importance_path)
    
    # Sort by importance in descending order and get the top features
    top_features = importance_df.sort_values(by='Importance', ascending=False).head(num_features)
    
    # Extract the feature indices (e.g., "Feature_97" -> 97)
    top_feature_indices = [int(f.split('_')[1]) for f in top_features['Feature']]
    
    return top_feature_indices

def train_and_evaluate(X_train, y_train, X_test, y_test, train_ids, test_ids,
                       class_names, output_dir, data_source_name, num_hidden=50,
                       learning_rate=0.01, num_epochs=100, batch_size=32, reg=0.01,
                       activation='sigmoid'):
    """
    Train neural network and evaluate on test set.
    
    Args:
        X_train: Training features
        y_train: Training labels (encoded)
        X_test: Test features
        y_test: Test labels (encoded)
        train_ids: Training sample IDs
        test_ids: Test sample IDs
        class_names: List of class names
        output_dir: Directory to save results
        data_source_name: Name for this data source (for file naming)
        num_hidden: Number of hidden units
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        reg: Regularization strength
    """
    num_classes = len(class_names)
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Convert labels to one-hot
    y_train_onehot = one_hot_labels(y_train, num_classes)
    y_test_onehot = one_hot_labels(y_test, num_classes)
    
    # Split training data into train and dev sets (80/20)
    n_train = len(X_train)
    n_dev = int(0.2 * n_train)
    indices = np.random.permutation(n_train)
    
    dev_indices = indices[:n_dev]
    train_indices = indices[n_dev:]
    
    X_train_split = X_train[train_indices]
    y_train_split_onehot = y_train_onehot[train_indices]
    X_dev = X_train[dev_indices]
    y_dev_onehot = y_train_onehot[dev_indices]
    
    print(f"\nTraining set: {len(X_train_split)} samples")
    print(f"Dev set: {len(X_dev)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    data_source_display = data_source_name.replace('_', ' ')
    print("\n" + "="*50)
    print(f"Training Neural Network - {data_source_display}")
    print("="*50)
    
    # Select activation function
    if activation.lower() == 'relu':
        act_func = relu
        act_derivative = relu_derivative
        activation_name = "ReLU"
    else:  # default to sigmoid
        act_func = sigmoid
        act_derivative = None  # sigmoid derivative computed from output
        activation_name = "Sigmoid"
    
    print(f"Model Configuration:")
    print(f"  Activation function: {activation_name}")
    print(f"  Hidden units: {num_hidden}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Regularization (L2): {reg}")
    
    # Create wrapper functions with activation function
    def forward_prop_wrapper(data, labels, params):
        return forward_prop(data, labels, params, activation_func=act_func)
    
    # Use regularized backward prop if reg > 0
    if reg > 0:
        def backward_prop_wrapper(data, labels, params, forward_func):
            return backward_prop_regularized(data, labels, params, forward_func, reg, activation_derivative=act_derivative)
    else:
        def backward_prop_wrapper(data, labels, params, forward_func):
            return backward_prop(data, labels, params, forward_func, activation_derivative=act_derivative)
    
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        X_train_split, y_train_split_onehot, X_dev, y_dev_onehot,
        get_initial_params, forward_prop_wrapper, backward_prop_wrapper,
        num_hidden=num_hidden, learning_rate=learning_rate, num_epochs=num_epochs,
        batch_size=batch_size, reg=reg, verbose=True,
        activation_func=act_func, activation_derivative=act_derivative,
        early_stopping=True, patience=15
    )
    
    # Format data source name for display
    data_source_display = data_source_name.replace('_', ' ')
    
    # Evaluate on full training set
    print("\n" + "="*50)
    print(f"Training Set Results - Neural Network - {data_source_display}")
    print("="*50)
    _, train_output, _ = forward_prop_wrapper(X_train, y_train_onehot, params)
    train_predictions = np.argmax(train_output, axis=1)
    
    # Compute both exact and grouped accuracy
    train_accuracy_exact = compute_accuracy(train_output, y_train_onehot)
    
    print(f"Training Accuracy (Exact Match): {train_accuracy_exact:.4f}")
    
    # Per-class accuracy on training set
    print(f"\nPer-class accuracy (Training - {data_source_display}):")
    for i, class_name in enumerate(class_names):
        mask = y_train == i
        if np.sum(mask) > 0:
            class_acc_exact = np.mean(train_predictions[mask] == y_train[mask])
            # Grouped accuracy for this class
            class_preds = train_predictions[mask]
            class_true = y_train[mask]
            print(f"  {class_name}: Exact={class_acc_exact:.4f}, ({np.sum(mask)} samples)")
    
    # Evaluate on test set
    if len(X_test) > 0:
        print("\n" + "="*50)
        print(f"Test Set Results - Neural Network - {data_source_display}")
        print("="*50)
        _, test_output, _ = forward_prop_wrapper(X_test, y_test_onehot, params)
        test_predictions = np.argmax(test_output, axis=1)
        test_probs = test_output
        
        # Compute both exact and grouped accuracy
        test_accuracy_exact = compute_accuracy(test_output, y_test_onehot)
        
        print(f"Test Accuracy (Exact Match): {test_accuracy_exact:.4f}")
        
        # Save test predictions (matching logreg format)
        test_pred_path = output_dir / f'test_predictions_{data_source_name}.txt'
        with open(test_pred_path, 'w') as f:
            f.write("Sample,True_Label,Predicted_Label,Confidence\n")
            for i in range(len(y_test)):
                true_label = idx_to_label[y_test[i]]
                pred_label = idx_to_label[test_predictions[i]]
                confidence = test_probs[i, test_predictions[i]]
                f.write(f"{i},{true_label},{pred_label},{confidence:.4f}\n")
        print(f"Test predictions saved to {test_pred_path}")
        
        # Plot confusion matrix for test set
        cm_test_path = output_dir / f'neural_network_confusion_matrix_{data_source_name}.png'
        plot_confusion_matrix(y_test, test_predictions, class_names, str(cm_test_path), 
                            title_suffix=data_source_display)
        
        # Per-class accuracy
        print(f"\nPer-class accuracy (Test - {data_source_display}):")
        for i, class_name in enumerate(class_names):
            mask = y_test == i
            if np.sum(mask) > 0:
                class_acc_exact = np.mean(test_predictions[mask] == y_test[mask])
                # Grouped accuracy for this class
                class_preds = test_predictions[mask]
                class_true = y_test[mask]
                print(f"  {class_name}: Exact={class_acc_exact:.4f}, ({np.sum(mask)} samples)")
        
        # Save accuracy summary
        accuracy_summary_path = output_dir / f'neural_network_accuracy_summary_{data_source_name}.txt'
        with open(accuracy_summary_path, 'w') as f:
            f.write(f"Neural Network Results - {data_source_display}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: Neural Network\n")
            f.write(f"Activation Function: {activation_name}\n")
            f.write(f"Data Source: {data_source_display}\n")
            f.write(f"Training Accuracy (Exact Match): {train_accuracy_exact:.4f}\n")
            f.write(f"Test Accuracy (Exact Match): {test_accuracy_exact:.4f}\n")
            f.write("Per-class Test Accuracy:\n")
            for i, class_name in enumerate(class_names):
                mask = y_test == i
                if np.sum(mask) > 0:
                    class_acc_exact = np.mean(test_predictions[mask] == y_test[mask])
                    class_preds = test_predictions[mask]
                    class_true = y_test[mask]
                    f.write(f"  {class_name}: Exact={class_acc_exact:.4f}, ({np.sum(mask)} samples)\n")
        print(f"Accuracy summary saved to {accuracy_summary_path}")
    
    return params


def main():
    """
    Main function to train and evaluate neural network on tennis serve classification.
    """
    parser = argparse.ArgumentParser(description='Train neural network for tennis serve classification')
    parser.add_argument('--num_hidden', type=int, default=50, help='Number of hidden units (reduced to prevent overfitting)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--reg', type=float, default=0.01, help='Regularization strength (L2)')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters for k-means features')
    parser.add_argument('--activation', type=str, default='sigmoid', choices=['sigmoid', 'relu'],
                        help='Activation function for hidden layer (sigmoid or relu, default: sigmoid)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Paths
    project_dir = Path(__file__).parent.parent.parent
    keypoints_dir = project_dir / 'data' / 'keypoints'
    labels_path = project_dir / 'data' / 'mcp_data.csv'
    output_dir = project_dir / 'src' / 'neuralnets' / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define class names
    class_names = ['4', '5', '6']
    num_classes = len(class_names)
    
    # Load labels
    print("Loading labels from CSV...")
    df = pd.read_csv(labels_path)
    print(f"Total samples in CSV: {len(df)}")
    
    # Split based on "Test Set" column
    print("CSV columns:", df.columns.tolist())
    df_test = df[df['Test Set'] == 'Test'].copy()
    df_train = df[df['Test Set'] != 'Test'].copy()
    #df_train['Serve Type'] = df_train['Serve Type'].apply(map_to_base_label)
    df_test['Serve Result'] = df_test['Serve Result'].apply(map_to_base_label)
    
    print(f"Training samples in CSV: {len(df_train)}")
    print(f"Test samples in CSV: {len(df_test)}")
    
    # ==========================================
    # Approach 1: Original JSON data (same as logreg)
    # ==========================================
    print("\n" + "="*70)
    print("APPROACH 1: Original JSON Keypoints Data")
    print("="*70)
    
    # Load keypoints data
    print("\nLoading keypoints data for training set...")
    X_train_json, y_train_json, train_ids_json = load_keypoints_data(str(keypoints_dir), df_train)
    print(f"Loaded {len(X_train_json)} training samples with {X_train_json.shape[1]} features each")
    
    print("\nLoading keypoints data for test set...")
    X_test_json, y_test_json, test_ids_json = load_keypoints_data(str(keypoints_dir), df_test)
    print(f"Loaded {len(X_test_json)} test samples")
    
    if len(X_train_json) == 0:
        print("Error: No training data loaded!")
        return
    
    # Encode labels
    y_train_json = [map_to_base_label(label) for label in y_train_json]
    y_train_json_enc = np.array([class_names.index(label) for label in y_train_json])
    if len(X_test_json) > 0:
        y_test_json = [map_to_base_label(label) for label in y_test_json]
        y_test_json_enc = np.array([class_names.index(label) for label in y_test_json])
    else:
        y_test_json_enc = np.array([])
        
    # Normalize features
    print("\nNormalizing features...")
    mean_json = np.mean(X_train_json, axis=0)
    std_json = np.std(X_train_json, axis=0)
    std_json[std_json == 0] = 1
    
    X_train_json_norm = (X_train_json - mean_json) / std_json
    if len(X_test_json) > 0:
        X_test_json_norm = (X_test_json - mean_json) / std_json
    else:
        X_test_json_norm = np.array([]).reshape(0, X_train_json.shape[1])
    
    # Train and evaluate on JSON data
    train_and_evaluate(
        X_train_json_norm, y_train_json_enc, X_test_json_norm, y_test_json_enc,
        train_ids_json, test_ids_json, class_names, output_dir, "JSON_Data",
        num_hidden=args.num_hidden, learning_rate=args.learning_rate,
        num_epochs=args.num_epochs, batch_size=args.batch_size, reg=args.reg,
        activation=args.activation
    )
    
    # ==========================================
    # Approach 2: K-means cluster features
    # ==========================================
    print("\n" + "="*70)
    print("APPROACH 2: K-Means Cluster Features")
    print("="*70)
    
    # Load clustering results
    clustering_results_path = project_dir / 'src' / 'k_means' / f'clustering_results_pose_k{args.num_clusters}.txt'
    
    print(f"\nLoading cluster assignments from {clustering_results_path}...")
    cluster_dict = load_cluster_assignments(str(clustering_results_path), train_ids_json + test_ids_json)
    print(f"Loaded cluster assignments for {len(cluster_dict)} samples")
    
    if len(cluster_dict) == 0:
        print("Warning: No cluster assignments found. Please run clustering.py first.")
        print("Skipping k-means approach.")
    else:
        # Add cluster features to JSON data
        X_train_cluster = add_cluster_features(X_train_json_norm, train_ids_json, cluster_dict, args.num_clusters)
        if len(X_test_json) > 0:
            X_test_cluster = add_cluster_features(X_test_json_norm, test_ids_json, cluster_dict, args.num_clusters)
        else:
            X_test_cluster = np.array([]).reshape(0, X_train_cluster.shape[1])
        
        print(f"Added {args.num_clusters} cluster features. New feature dimension: {X_train_cluster.shape[1]}")
        
        # Train and evaluate on cluster-enhanced data
        train_and_evaluate(
            X_train_cluster, y_train_json_enc, X_test_cluster, y_test_json_enc,
            train_ids_json, test_ids_json, class_names, output_dir, "K_Means_Cluster_Data",
            num_hidden=args.num_hidden, learning_rate=args.learning_rate,
            num_epochs=args.num_epochs, batch_size=args.batch_size, reg=args.reg,
            activation=args.activation
        )
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == '__main__':
    main()

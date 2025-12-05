import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys

# Add parent directory to path to import from logreg
sys.path.append(str(Path(__file__).parent.parent))


def load_sequence_data(keypoints_dir, labels_df):
    """
    Load keypoints data as sequences (not aggregated) for GRU processing.
    
    Args:
        keypoints_dir: Path to directory containing keypoints subdirectories
        labels_df: DataFrame with labels (already filtered for training/validation)
    
    Returns:
        X: List of sequences, each is (n_frames, 75) numpy array
        y: Label array (n_samples,)
        sample_ids: List of sample identifiers
        sequence_lengths: List of actual sequence lengths
    """
    X_list = []
    y_list = []
    sample_ids = []
    sequence_lengths = []
    
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
        
        # Extract features from all frames (keep as sequence)
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
        
        # Keep as sequence (n_frames, 75)
        frame_features = np.array(frame_features)  # shape: (n_frames, 75)
        
        X_list.append(frame_features)
        y_list.append(serve_result)
        sample_ids.append(dir_name)
        sequence_lengths.append(len(frame_features))
    
    return X_list, np.array(y_list), sample_ids, sequence_lengths


def map_label_to_direction(label):
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


def pad_sequences(sequences, max_length=None, pad_value=0.0):
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of numpy arrays of shape (n_frames, feature_dim)
        max_length: Maximum sequence length. If None, use the longest sequence.
        pad_value: Value to use for padding
    
    Returns:
        padded_sequences: Numpy array of shape (n_samples, max_length, feature_dim)
        sequence_lengths: Array of actual sequence lengths
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    feature_dim = sequences[0].shape[1]
    n_samples = len(sequences)
    
    padded = np.full((n_samples, max_length, feature_dim), pad_value, dtype=np.float32)
    lengths = np.zeros(n_samples, dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        length = len(seq)
        lengths[i] = length
        if length > max_length:
            # Truncate if too long
            padded[i] = seq[:max_length]
        else:
            padded[i, :length] = seq
    
    return padded, lengths


class GRUCell:
    """
    GRU (Gated Recurrent Unit) Cell implementation from scratch.
    """
    
    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights using Xavier initialization
        scale = np.sqrt(1.0 / input_size)
        self.W_r = np.random.normal(0, scale, (input_size, hidden_size))  # Reset gate
        self.W_z = np.random.normal(0, scale, (input_size, hidden_size))  # Update gate
        self.W_h = np.random.normal(0, scale, (input_size, hidden_size))  # Candidate hidden state
        
        scale_h = np.sqrt(1.0 / hidden_size)
        self.U_r = np.random.normal(0, scale_h, (hidden_size, hidden_size))  # Reset gate
        self.U_z = np.random.normal(0, scale_h, (hidden_size, hidden_size))  # Update gate
        self.U_h = np.random.normal(0, scale_h, (hidden_size, hidden_size))  # Candidate hidden state
        
        # Bias terms
        self.b_r = np.zeros(hidden_size)
        self.b_z = np.zeros(hidden_size)
        self.b_h = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function."""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return np.tanh(x)
    
    def forward(self, x, h_prev):
        """
        Forward pass through GRU cell.
        
        Args:
            x: Input at current time step (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
        
        Returns:
            h: New hidden state (batch_size, hidden_size)
            cache: Dictionary storing intermediate values for backprop
        """
        # Reset gate
        r = self.sigmoid(x @ self.W_r + h_prev @ self.U_r + self.b_r)
        
        # Update gate
        z = self.sigmoid(x @ self.W_z + h_prev @ self.U_z + self.b_z)
        
        # Candidate hidden state
        h_tilde = self.tanh(x @ self.W_h + (r * h_prev) @ self.U_h + self.b_h)
        
        # Hidden state update
        h = (1 - z) * h_prev + z * h_tilde
        
        cache = {
            'x': x, 'h_prev': h_prev, 'r': r, 'z': z, 'h_tilde': h_tilde, 'h': h
        }
        
        return h, cache
    
    def backward(self, dh, cache):
        """
        Backward pass through GRU cell.
        
        Args:
            dh: Gradient w.r.t. hidden state (batch_size, hidden_size)
            cache: Dictionary from forward pass
        
        Returns:
            dx: Gradient w.r.t. input (batch_size, input_size)
            dh_prev: Gradient w.r.t. previous hidden state (batch_size, hidden_size)
            grads: Dictionary of gradients for parameters
        """
        x = cache['x']
        h_prev = cache['h_prev']
        r = cache['r']
        z = cache['z']
        h_tilde = cache['h_tilde']
        h = cache['h']
        
        # Gradient through hidden state update: h = (1 - z) * h_prev + z * h_tilde
        dh_prev_from_update = dh * (1 - z)
        dz = dh * (-h_prev + h_tilde)
        dh_tilde = dh * z
        
        # Gradient through candidate hidden state: h_tilde = tanh(...)
        dh_tilde_raw = dh_tilde * (1 - h_tilde ** 2)
        
        # Gradient through reset gate in h_tilde computation
        dr_from_h_tilde = (dh_tilde_raw @ self.U_h.T) * h_prev
        dh_prev_from_h_tilde = (dh_tilde_raw @ self.U_h.T) * r
        
        # Total gradient through reset gate
        dr_total = dr_from_h_tilde
        dr_raw = dr_total * r * (1 - r)
        
        # Total gradient through update gate
        dz_raw = dz * z * (1 - z)
        
        # Gradients w.r.t. inputs
        dx = (dr_raw @ self.W_r.T + dz_raw @ self.W_z.T + dh_tilde_raw @ self.W_h.T)
        
        dh_prev = (dh_prev_from_update + 
                   dr_raw @ self.U_r.T + 
                   dz_raw @ self.U_z.T + 
                   dh_prev_from_h_tilde)
        
        # Parameter gradients (average over batch)
        batch_size = x.shape[0]
        grads = {
            'W_r': (x.T @ dr_raw) / batch_size, 
            'U_r': (h_prev.T @ dr_raw) / batch_size, 
            'b_r': np.sum(dr_raw, axis=0) / batch_size,
            'W_z': (x.T @ dz_raw) / batch_size, 
            'U_z': (h_prev.T @ dz_raw) / batch_size, 
            'b_z': np.sum(dz_raw, axis=0) / batch_size,
            'W_h': (x.T @ dh_tilde_raw) / batch_size, 
            'U_h': ((r * h_prev).T @ dh_tilde_raw) / batch_size, 
            'b_h': np.sum(dh_tilde_raw, axis=0) / batch_size,
        }
        
        return dx, dh_prev, grads


class GRUClassifier:
    """
    GRU-based classifier for sequence classification.
    """
    
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        """
        Args:
            input_size: Size of input features (75 for keypoints)
            hidden_size: Size of GRU hidden state
            num_classes: Number of output classes (3: 4, 5, 6)
            num_layers: Number of GRU layers (default: 1)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Create GRU cells
        self.gru_cells = []
        for i in range(num_layers):
            if i == 0:
                cell_input_size = input_size
            else:
                cell_input_size = hidden_size
            self.gru_cells.append(GRUCell(cell_input_size, hidden_size))
        
        # Output layer (from hidden state to classes)
        scale = np.sqrt(1.0 / hidden_size)
        self.W_out = np.random.normal(0, scale, (hidden_size, num_classes))
        self.b_out = np.zeros(num_classes)
    
    def forward(self, X, sequence_lengths):
        """
        Forward pass through GRU.
        
        Args:
            X: Input sequences (batch_size, max_length, input_size)
            sequence_lengths: Actual lengths of each sequence (batch_size,)
        
        Returns:
            output: Class probabilities (batch_size, num_classes)
            cache: Dictionary for backprop
        """
        batch_size, max_length, _ = X.shape
        
        # Store caches organized by layer and time step
        # caches[layer_idx][t] = cache for layer at time t
        caches = [[None] * max_length for _ in range(self.num_layers)]
        
        # Initialize hidden states for each layer
        h_states = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        
        # Process each time step
        for t in range(max_length):
            x_t = X[:, t, :]  # (batch_size, input_size)
            
            # Pass through GRU layers
            for layer_idx, gru_cell in enumerate(self.gru_cells):
                h_prev = h_states[layer_idx]
                
                h, cache = gru_cell.forward(x_t, h_prev)
                
                # Store cache
                caches[layer_idx][t] = cache
                
                # Update states only for valid time steps
                # For padded time steps, keep the previous state
                valid_mask = (sequence_lengths > t).astype(float).reshape(-1, 1)
                h_states[layer_idx] = h * valid_mask + h_states[layer_idx] * (1 - valid_mask)
                
                # Output of one layer is input to next
                x_t = h
        
        # Use final hidden state from last layer for classification
        # This is the state at the last valid time step for each sequence
        final_h = h_states[-1]
        
        # Apply output layer
        logits = final_h @ self.W_out + self.b_out  # (batch_size, num_classes)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs, {'caches': caches, 'final_h': final_h, 'logits': logits, 
                      'h_states': h_states, 'max_length': max_length}
    
    def backward(self, probs, y_onehot, cache, sequence_lengths):
        """
        Backward pass through GRU using Backpropagation Through Time (BPTT).
        
        Args:
            probs: Predicted probabilities (batch_size, num_classes)
            y_onehot: One-hot encoded true labels (batch_size, num_classes)
            cache: Dictionary from forward pass
            sequence_lengths: Actual lengths of each sequence
        
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        caches = cache['caches']
        final_h = cache['final_h']
        logits = cache['logits']
        max_length = cache['max_length']
        batch_size = probs.shape[0]
        
        # Gradient of cross-entropy loss w.r.t. logits: dL/dlogits = probs - y_onehot
        # This is the standard result for cross-entropy + softmax
        # Note: We don't divide by batch_size here because the loss is already averaged
        dlogits = probs - y_onehot
        
        # Gradient through output layer
        dh_final = dlogits @ self.W_out.T  # (batch_size, hidden_size)
        dW_out = (final_h.T @ dlogits) / batch_size  # Average over batch
        db_out = np.sum(dlogits, axis=0) / batch_size  # Average over batch
        
        # Initialize gradients for GRU cells
        grads = {
            'W_out': dW_out,
            'b_out': db_out
        }
        
        # Initialize cell gradients
        for layer_idx in range(self.num_layers):
            cell = self.gru_cells[layer_idx]
            grads[f'W_r_{layer_idx}'] = np.zeros_like(cell.W_r)
            grads[f'U_r_{layer_idx}'] = np.zeros_like(cell.U_r)
            grads[f'b_r_{layer_idx}'] = np.zeros_like(cell.b_r)
            grads[f'W_z_{layer_idx}'] = np.zeros_like(cell.W_z)
            grads[f'U_z_{layer_idx}'] = np.zeros_like(cell.U_z)
            grads[f'b_z_{layer_idx}'] = np.zeros_like(cell.b_z)
            grads[f'W_h_{layer_idx}'] = np.zeros_like(cell.W_h)
            grads[f'U_h_{layer_idx}'] = np.zeros_like(cell.U_h)
            grads[f'b_h_{layer_idx}'] = np.zeros_like(cell.b_h)
        
        # Backpropagation Through Time (BPTT)
        # Process each layer separately, starting from the last layer
        dx_next_layer = None
        
        for layer_idx in reversed(range(self.num_layers)):
            # Initialize gradients for this layer
            dh = np.zeros((batch_size, self.hidden_size))
            
            # For the last layer, add gradient from output
            # Only apply gradient to sequences at their last valid time step
            if layer_idx == self.num_layers - 1:
                dh = dh_final.copy()
            else:
                # For intermediate layers, use gradient from next layer
                dh = dx_next_layer.copy()
            
            cell = self.gru_cells[layer_idx]
            
            # Backpropagate through time (reverse order)
            # Start from the last valid time step for each sequence
            for t in reversed(range(max_length)):
                cell_cache = caches[layer_idx][t]
                if cell_cache is None:
                    continue
                
                # Only backpropagate through valid time steps (respect sequence lengths)
                # For padded time steps (t >= sequence_length), zero out gradients
                valid_mask = (sequence_lengths > t).astype(float).reshape(-1, 1)
                
                # Mask gradients going into backward pass
                dh_masked = dh * valid_mask
                
                dx, dh_prev, cell_grads = cell.backward(dh_masked, cell_cache)
                
                # Accumulate parameter gradients (only from valid time steps)
                grads[f'W_r_{layer_idx}'] += cell_grads['W_r']
                grads[f'U_r_{layer_idx}'] += cell_grads['U_r']
                grads[f'b_r_{layer_idx}'] += cell_grads['b_r']
                grads[f'W_z_{layer_idx}'] += cell_grads['W_z']
                grads[f'U_z_{layer_idx}'] += cell_grads['U_z']
                grads[f'b_z_{layer_idx}'] += cell_grads['b_z']
                grads[f'W_h_{layer_idx}'] += cell_grads['W_h']
                grads[f'U_h_{layer_idx}'] += cell_grads['U_h']
                grads[f'b_h_{layer_idx}'] += cell_grads['b_h']
                
                # Pass gradients to previous time step
                # For valid sequences, use the computed gradients; for padded, keep zero
                dh = dh_prev * valid_mask
            
            # For multi-layer, pass dx to previous layer
            dx_next_layer = dx
        
        return grads
    
    def compute_loss(self, probs, y_onehot):
        """
        Compute cross-entropy loss.
        
        Args:
            probs: Predicted probabilities (batch_size, num_classes)
            y_onehot: One-hot encoded labels (batch_size, num_classes)
        
        Returns:
            loss: Average cross-entropy loss
        """
        epsilon = 1e-10
        loss = -np.mean(np.sum(y_onehot * np.log(probs + epsilon), axis=1))
        return loss
    
    def update_params(self, grads, learning_rate, clip_value=5.0):
        """
        Update parameters using gradients with gradient clipping.
        
        Args:
            grads: Dictionary of gradients
            learning_rate: Learning rate
            clip_value: Maximum gradient norm for clipping
        """
        # Clip gradients to prevent exploding gradients
        total_norm = 0.0
        for key, grad in grads.items():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > clip_value:
            clip_coef = clip_value / (total_norm + 1e-6)
            for key in grads:
                grads[key] *= clip_coef
        
        # Update output layer
        self.W_out -= learning_rate * grads['W_out']
        self.b_out -= learning_rate * grads['b_out']
        
        # Update GRU cells
        for layer_idx in range(self.num_layers):
            cell = self.gru_cells[layer_idx]
            cell.W_r -= learning_rate * grads[f'W_r_{layer_idx}']
            cell.U_r -= learning_rate * grads[f'U_r_{layer_idx}']
            cell.b_r -= learning_rate * grads[f'b_r_{layer_idx}']
            cell.W_z -= learning_rate * grads[f'W_z_{layer_idx}']
            cell.U_z -= learning_rate * grads[f'U_z_{layer_idx}']
            cell.b_z -= learning_rate * grads[f'b_z_{layer_idx}']
            cell.W_h -= learning_rate * grads[f'W_h_{layer_idx}']
            cell.U_h -= learning_rate * grads[f'U_h_{layer_idx}']
            cell.b_h -= learning_rate * grads[f'b_h_{layer_idx}']


def one_hot_encode(labels, num_classes):
    """
    Convert class indices to one-hot encoding.
    
    Args:
        labels: Array of class indices (batch_size,)
        num_classes: Number of classes
    
    Returns:
        one_hot: One-hot encoded labels (batch_size, num_classes)
    """
    batch_size = len(labels)
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), labels] = 1
    return one_hot


def compute_accuracy(predictions, true_labels):
    """
    Compute accuracy.
    
    Args:
        predictions: Predicted class indices (batch_size,)
        true_labels: True class indices (batch_size,)
    
    Returns:
        accuracy: Fraction of correct predictions
    """
    return np.mean(predictions == true_labels)


def train_gru(model, X_train, y_train, X_dev, y_dev, sequence_lengths_train, 
              sequence_lengths_dev, learning_rate=0.01, num_epochs=50, 
              batch_size=16, verbose=True):
    """
    Train GRU model.
    
    Args:
        model: GRUClassifier instance
        X_train: Training sequences (n_train, max_length, input_size)
        y_train: Training labels (n_train,)
        X_dev: Dev sequences (n_dev, max_length, input_size)
        y_dev: Dev labels (n_dev,)
        sequence_lengths_train: Training sequence lengths
        sequence_lengths_dev: Dev sequence lengths
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        verbose: Whether to print progress
    
    Returns:
        train_losses: List of training losses
        dev_losses: List of dev losses
        train_accuracies: List of training accuracies
        dev_accuracies: List of dev accuracies
    """
    n_train = len(X_train)
    train_losses = []
    dev_losses = []
    train_accuracies = []
    dev_accuracies = []
    
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = np.random.permutation(n_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        seq_lens_shuffled = sequence_lengths_train[indices]
        
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            X_batch = X_train_shuffled[i:end_idx]
            y_batch = y_train_shuffled[i:end_idx]
            seq_lens_batch = seq_lens_shuffled[i:end_idx]
            
            # Forward pass
            probs, cache = model.forward(X_batch, seq_lens_batch)
            y_onehot = one_hot_encode(y_batch, model.num_classes)
            loss = model.compute_loss(probs, y_onehot)
            epoch_losses.append(loss)
            
            # Backward pass
            # Gradient of cross-entropy loss: dL/dlogits = probs - y_onehot
            grads = model.backward(probs, y_onehot, cache, seq_lens_batch)
            
            # Update parameters
            model.update_params(grads, learning_rate)
        
        # Evaluate on training set
        train_probs, _ = model.forward(X_train, sequence_lengths_train)
        train_preds = np.argmax(train_probs, axis=1)
        train_acc = compute_accuracy(train_preds, y_train)
        train_loss = model.compute_loss(train_probs, one_hot_encode(y_train, model.num_classes))
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate on dev set
        if len(X_dev) > 0:
            dev_probs, _ = model.forward(X_dev, sequence_lengths_dev)
            dev_preds = np.argmax(dev_probs, axis=1)
            dev_acc = compute_accuracy(dev_preds, y_dev)
            dev_loss = model.compute_loss(dev_probs, one_hot_encode(y_dev, model.num_classes))
            
            dev_losses.append(dev_loss)
            dev_accuracies.append(dev_acc)
        
        if verbose and (epoch % 5 == 0 or epoch == num_epochs - 1):
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}', end='')
            if len(X_dev) > 0:
                print(f', Dev Loss = {dev_loss:.4f}, Dev Acc = {dev_acc:.4f}')
            else:
                print()
    
    return train_losses, dev_losses, train_accuracies, dev_accuracies


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        class_names: List of class names
        save_path: Path to save the plot
    """
    num_classes = len(class_names)
    all_labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted label',
           ylabel='True label',
           title='Confusion Matrix - GRU')
    
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


def main():
    """
    Main function to train and evaluate GRU model for tennis serve classification.
    Randomly splits 100 examples into train (70), dev (15), and test (15).
    """
    # Set random seed for reproducibility
    seed = 12
    np.random.seed(seed)
    
    # Paths
    project_dir = Path(__file__).parent.parent.parent
    keypoints_dir = project_dir / 'data' / 'keypoints'
    labels_path = project_dir / 'data' / 'mcp_data.csv'
    output_dir = project_dir / 'src' / 'gru' / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print("Loading labels from CSV...")
    df = pd.read_csv(labels_path)
    print(f"Total samples in CSV: {len(df)}")
    
    # Load all sequence data first
    print("\nLoading all sequence data...")
    X_all_seq, y_all_raw, all_ids, seq_lens_all = load_sequence_data(
        str(keypoints_dir), df)
    print(f"Loaded {len(X_all_seq)} total sequences")
    
    if len(X_all_seq) == 0:
        print("Error: No data loaded!")
        return
    
    print(f"Sequence lengths - Min: {min(seq_lens_all)}, Max: {max(seq_lens_all)}, "
          f"Mean: {np.mean(seq_lens_all):.1f}")
    
    # Map labels to 3 classes (4, 5, 6)
    print("\nMapping labels to 3 classes (4, 5, 6)...")
    direction_classes = ['4', '5', '6']
    direction_to_idx = {d: i for i, d in enumerate(direction_classes)}
    
    # Convert labels to strings if they're not already (handles numeric labels)
    y_all_raw_str = [str(label) for label in y_all_raw]
    y_all = np.array([direction_to_idx[map_label_to_direction(label)] for label in y_all_raw_str])
    
    print(f"Label distribution (all data):")
    for i, cls in enumerate(direction_classes):
        count = np.sum(y_all == i)
        print(f"  {cls}: {count} samples")
    
    # Randomly split into train (70), dev (15), test (15)
    n_total = len(X_all_seq)
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
    X_train_seq = [X_all_seq[i] for i in train_indices]
    y_train_raw = [y_all_raw_str[i] for i in train_indices]
    seq_lens_train = [seq_lens_all[i] for i in train_indices]
    y_train = y_all[train_indices]
    
    X_dev_seq = [X_all_seq[i] for i in dev_indices]
    y_dev_raw = [y_all_raw_str[i] for i in dev_indices]
    seq_lens_dev = [seq_lens_all[i] for i in dev_indices]
    y_dev = y_all[dev_indices]
    
    X_test_seq = [X_all_seq[i] for i in test_indices]
    y_test_raw = [y_all_raw_str[i] for i in test_indices]
    seq_lens_test = [seq_lens_all[i] for i in test_indices]
    y_test = y_all[test_indices]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train_seq)} samples")
    print(f"  Dev/Eval: {len(X_dev_seq)} samples")
    print(f"  Test: {len(X_test_seq)} samples")
    
    # Normalize features (compute stats from training data only)
    print("\nNormalizing features...")
    # Flatten all training sequences to compute statistics
    all_train_features = np.concatenate(X_train_seq, axis=0)
    mean = np.mean(all_train_features, axis=0)
    std = np.std(all_train_features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    # Normalize each sequence
    X_train_norm = [(seq - mean) / std for seq in X_train_seq]
    X_dev_norm = [(seq - mean) / std for seq in X_dev_seq]
    X_test_norm = [(seq - mean) / std for seq in X_test_seq]
    
    # Pad sequences to same length
    print("\nPadding sequences...")
    max_length = max(seq_lens_train + seq_lens_dev + seq_lens_test)
    
    X_train_padded, seq_lens_train_arr = pad_sequences(X_train_norm, max_length=max_length)
    X_dev_padded, seq_lens_dev_arr = pad_sequences(X_dev_norm, max_length=max_length)
    X_test_padded, seq_lens_test_arr = pad_sequences(X_test_norm, max_length=max_length)
    
    print(f"Padded sequences to length: {max_length}")
    print(f"Training data shape: {X_train_padded.shape}")
    print(f"Dev data shape: {X_dev_padded.shape}")
    print(f"Test data shape: {X_test_padded.shape}")
    
    # Convert to numpy arrays
    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    y_test = np.array(y_test)
    seq_lens_train_arr = np.array(seq_lens_train_arr)
    seq_lens_dev_arr = np.array(seq_lens_dev_arr)
    seq_lens_test_arr = np.array(seq_lens_test_arr)
    
    # Create and train GRU model
    print("\n" + "="*50)
    print("Training GRU Model")
    print("="*50)
    
    input_size = X_train_padded.shape[2]  # 75 keypoints
    hidden_size = 64
    num_classes = 3  # 4, 5, 6
    
    model = GRUClassifier(input_size=input_size, hidden_size=hidden_size, 
                         num_classes=num_classes, num_layers=1)
    
    print(f"Model configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Max sequence length: {max_length}")
    
    # Train model
    train_losses, dev_losses, train_accuracies, dev_accuracies = train_gru(
        model, X_train_padded, y_train, X_dev_padded, y_dev,
        seq_lens_train_arr, seq_lens_dev_arr,
        learning_rate=0.01, num_epochs=50, batch_size=16, verbose=True
    )
    
    # Evaluate on training set
    print("\n" + "="*50)
    print("Training Set Results")
    print("="*50)
    train_probs, _ = model.forward(X_train_padded, seq_lens_train_arr)
    train_preds = np.argmax(train_probs, axis=1)
    train_acc = compute_accuracy(train_preds, y_train)
    print(f"Training Accuracy: {train_acc:.4f}")
    
    # Per-class accuracy on training set
    print("\nPer-class accuracy (Training):")
    for i, class_name in enumerate(direction_classes):
        mask = y_train == i
        if np.sum(mask) > 0:
            class_acc = np.mean(train_preds[mask] == y_train[mask])
            print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # Evaluate on dev set
    print("\n" + "="*50)
    print("Dev/Eval Set Results")
    print("="*50)
    dev_probs, _ = model.forward(X_dev_padded, seq_lens_dev_arr)
    dev_preds = np.argmax(dev_probs, axis=1)
    dev_acc = compute_accuracy(dev_preds, y_dev)
    print(f"Dev Accuracy: {dev_acc:.4f}")
    
    # Per-class accuracy on dev set
    print("\nPer-class accuracy (Dev):")
    for i, class_name in enumerate(direction_classes):
        mask = y_dev == i
        if np.sum(mask) > 0:
            class_acc = np.mean(dev_preds[mask] == y_dev[mask])
            print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # Evaluate on test set
    if len(X_test_padded) > 0:
        print("\n" + "="*50)
        print("Test Set Results")
        print("="*50)
        test_probs, _ = model.forward(X_test_padded, seq_lens_test_arr)
        test_preds = np.argmax(test_probs, axis=1)
        test_acc = compute_accuracy(test_preds, y_test)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Per-class accuracy
        print("\nPer-class accuracy (Test):")
        for i, class_name in enumerate(direction_classes):
            mask = y_test == i
            if np.sum(mask) > 0:
                class_acc = np.mean(test_preds[mask] == y_test[mask])
                print(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
        
        # Save test predictions
        test_pred_path = output_dir / f'gru_test_predictions_seed_{seed}.txt'
        with open(test_pred_path, 'w') as f:
            f.write("Sample,True_Label,Predicted_Label,Confidence\n")
            for i in range(len(y_test)):
                true_label = direction_classes[y_test[i]]
                pred_label = direction_classes[test_preds[i]]
                confidence = test_probs[i, test_preds[i]]
                f.write(f"{i},{true_label},{pred_label},{confidence:.4f}\n")
        print(f"\nTest predictions saved to {test_pred_path}")
        
        # Plot confusion matrix
        cm_path = output_dir / f'gru_confusion_matrix_seed_{seed}.png'
        plot_confusion_matrix(y_test, test_preds, direction_classes, str(cm_path))
        
        # Save accuracy summary
        summary_path = output_dir / f'gru_accuracy_summary_seed_{seed}.txt'
        with open(summary_path, 'w') as f:
            f.write("GRU Model Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data Split: Train=70, Dev=15, Test=15\n\n")
            f.write(f"Training Accuracy: {train_acc:.4f}\n")
            f.write(f"Dev Accuracy: {dev_acc:.4f}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
            f.write("Per-class Test Accuracy:\n")
            for i, class_name in enumerate(direction_classes):
                mask = y_test == i
                if np.sum(mask) > 0:
                    class_acc = np.mean(test_preds[mask] == y_test[mask])
                    f.write(f"  {class_name}: {class_acc:.4f} ({np.sum(mask)} samples)\n")
        print(f"Accuracy summary saved to {summary_path}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == '__main__':
    main()


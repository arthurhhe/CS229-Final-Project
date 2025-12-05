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
    Load keypoints data as sequences (not aggregated) for 1D CNN processing.
    
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


def conv1d_forward(x, w, b, stride=1, padding=0):
    """
    1D Convolution forward pass.
    
    Args:
        x: Input (batch_size, in_channels, sequence_length)
        w: Weights (out_channels, in_channels, kernel_size)
        b: Bias (out_channels,)
        stride: Stride size
        padding: Padding size
    
    Returns:
        out: Output (batch_size, out_channels, output_length)
        cache: Cache for backward pass
    """
    batch_size, in_channels, seq_len = x.shape
    out_channels, _, kernel_size = w.shape
    
    # Add padding
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')
    else:
        x_padded = x
    
    # Calculate output length
    output_length = (x_padded.shape[2] - kernel_size) // stride + 1
    
    # Initialize output
    out = np.zeros((batch_size, out_channels, output_length))
    
    # Perform convolution
    for i in range(output_length):
        start_idx = i * stride
        end_idx = start_idx + kernel_size
        x_slice = x_padded[:, :, start_idx:end_idx]  # (batch, in_channels, kernel_size)
        
        # Convolve: sum over in_channels and kernel_size dimensions
        for oc in range(out_channels):
            out[:, oc, i] = np.sum(x_slice * w[oc:oc+1, :, :], axis=(1, 2)) + b[oc]
    
    cache = (x, w, b, stride, padding)
    return out, cache


def conv1d_backward(dout, cache):
    """
    1D Convolution backward pass.
    
    Args:
        dout: Gradient w.r.t. output (batch_size, out_channels, output_length)
        cache: Cache from forward pass
    
    Returns:
        dx: Gradient w.r.t. input (batch_size, in_channels, sequence_length)
        dw: Gradient w.r.t. weights (out_channels, in_channels, kernel_size)
        db: Gradient w.r.t. bias (out_channels,)
    """
    x, w, b, stride, padding = cache
    batch_size, in_channels, seq_len = x.shape
    out_channels, _, kernel_size = w.shape
    
    # Add padding to input for gradient computation
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')
    else:
        x_padded = x
    
    output_length = dout.shape[2]
    
    # Initialize gradients
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Backward pass
    for i in range(output_length):
        start_idx = i * stride
        end_idx = start_idx + kernel_size
        x_slice = x_padded[:, :, start_idx:end_idx]  # (batch, in_channels, kernel_size)
        
        for oc in range(out_channels):
            # Gradient w.r.t. bias
            db[oc] += np.sum(dout[:, oc, i])
            
            # Gradient w.r.t. weights
            dw[oc] += np.sum(x_slice * dout[:, oc, i:i+1, np.newaxis], axis=0)
            
            # Gradient w.r.t. input
            dx_padded[:, :, start_idx:end_idx] += w[oc:oc+1, :, :] * dout[:, oc, i:i+1, np.newaxis]
    
    # Remove padding from dx
    if padding > 0:
        dx = dx_padded[:, :, padding:-padding]
    else:
        dx = dx_padded
    
    # Average over batch
    dw /= batch_size
    db /= batch_size
    
    return dx, dw, db


def max_pool1d_forward(x, pool_size, stride=None):
    """
    1D Max Pooling forward pass.
    
    Args:
        x: Input (batch_size, channels, sequence_length)
        pool_size: Pooling window size
        stride: Stride size (default: pool_size)
    
    Returns:
        out: Output (batch_size, channels, output_length)
        cache: Cache for backward pass
    """
    if stride is None:
        stride = pool_size
    
    batch_size, channels, seq_len = x.shape
    output_length = (seq_len - pool_size) // stride + 1
    
    out = np.zeros((batch_size, channels, output_length))
    argmax = np.zeros((batch_size, channels, output_length), dtype=np.int32)
    
    for i in range(output_length):
        start_idx = i * stride
        end_idx = start_idx + pool_size
        x_slice = x[:, :, start_idx:end_idx]  # (batch, channels, pool_size)
        
        out[:, :, i] = np.max(x_slice, axis=2)
        argmax[:, :, i] = np.argmax(x_slice, axis=2) + start_idx
    
    cache = (x, pool_size, stride, argmax)
    return out, cache


def max_pool1d_backward(dout, cache):
    """
    1D Max Pooling backward pass.
    
    Args:
        dout: Gradient w.r.t. output (batch_size, channels, output_length)
        cache: Cache from forward pass
    
    Returns:
        dx: Gradient w.r.t. input (batch_size, channels, sequence_length)
    """
    x, pool_size, stride, argmax = cache
    batch_size, channels, seq_len = x.shape
    
    dx = np.zeros_like(x)
    output_length = dout.shape[2]
    
    for i in range(output_length):
        for c in range(channels):
            for b in range(batch_size):
                dx[b, c, argmax[b, c, i]] += dout[b, c, i]
    
    return dx


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_backward(dout, x):
    """ReLU backward pass."""
    dx = dout * (x > 0)
    return dx


class Conv1DLayer:
    """1D Convolutional layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride size
            padding: Padding size
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights using Xavier initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.w = np.random.normal(0, scale, (out_channels, in_channels, kernel_size))
        self.b = np.zeros(out_channels)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input (batch_size, in_channels, sequence_length)
        
        Returns:
            out: Output (batch_size, out_channels, output_length)
        """
        out, cache = conv1d_forward(x, self.w, self.b, self.stride, self.padding)
        self.cache = cache
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Args:
            dout: Gradient w.r.t. output (batch_size, out_channels, output_length)
        
        Returns:
            dx: Gradient w.r.t. input (batch_size, in_channels, sequence_length)
        """
        dx, dw, db = conv1d_backward(dout, self.cache)
        self.dw = dw
        self.db = db
        return dx
    
    def update(self, learning_rate):
        """Update parameters."""
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db


class MaxPool1DLayer:
    """1D Max Pooling layer."""
    
    def __init__(self, pool_size, stride=None):
        """
        Args:
            pool_size: Pooling window size
            stride: Stride size (default: pool_size)
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input (batch_size, channels, sequence_length)
        
        Returns:
            out: Output (batch_size, channels, output_length)
        """
        out, cache = max_pool1d_forward(x, self.pool_size, self.stride)
        self.cache = cache
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Args:
            dout: Gradient w.r.t. output (batch_size, channels, output_length)
        
        Returns:
            dx: Gradient w.r.t. input (batch_size, channels, sequence_length)
        """
        dx = max_pool1d_backward(dout, self.cache)
        return dx


class DenseLayer:
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size: Size of input
            output_size: Size of output
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights using Xavier initialization
        scale = np.sqrt(1.0 / input_size)
        self.w = np.random.normal(0, scale, (input_size, output_size))
        self.b = np.zeros(output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input (batch_size, input_size)
        
        Returns:
            out: Output (batch_size, output_size)
        """
        self.x = x
        out = x @ self.w + self.b
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        
        Args:
            dout: Gradient w.r.t. output (batch_size, output_size)
        
        Returns:
            dx: Gradient w.r.t. input (batch_size, input_size)
        """
        batch_size = dout.shape[0]
        self.dw = (self.x.T @ dout) / batch_size
        self.db = np.sum(dout, axis=0) / batch_size
        dx = dout @ self.w.T
        return dx
    
    def update(self, learning_rate):
        """Update parameters."""
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db


class CNN1DClassifier:
    """
    1D CNN-based classifier for sequence classification.
    Uses multiple kernel sizes (3, 5, 7) to capture different temporal patterns.
    """
    
    def __init__(self, input_size, num_classes, num_filters=32, kernel_sizes=[3, 5, 7], hidden_size=64):
        """
        Args:
            input_size: Size of input features (75 for keypoints)
            num_classes: Number of output classes (3: 4, 5, 6)
            num_filters: Number of filters per kernel size
            kernel_sizes: List of kernel sizes to use
            hidden_size: Size of hidden layer in fully connected part
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.hidden_size = hidden_size
        
        # Create parallel 1D conv layers with different kernel sizes
        self.conv_layers = []
        for kernel_size in kernel_sizes:
            # Padding to maintain sequence length approximately
            padding = kernel_size // 2
            conv = Conv1DLayer(input_size, num_filters, kernel_size, stride=1, padding=padding)
            self.conv_layers.append(conv)
        
        # Pooling layer (global max pooling over time)
        self.pool = MaxPool1DLayer(pool_size=2, stride=2)
        
        # Calculate size after convolutions and pooling
        # Each conv outputs (batch, num_filters, seq_len), we'll use global max pooling
        # Total features = num_kernels * num_filters
        total_features = len(kernel_sizes) * num_filters
        
        # Fully connected layers
        self.fc1 = DenseLayer(total_features, hidden_size)
        self.fc2 = DenseLayer(hidden_size, num_classes)
    
    def forward(self, X, sequence_lengths):
        """
        Forward pass through 1D CNN.
        
        Args:
            X: Input sequences (batch_size, max_length, input_size)
            sequence_lengths: Actual lengths of each sequence (batch_size,)
        
        Returns:
            output: Class probabilities (batch_size, num_classes)
            cache: Dictionary for backprop
        """
        batch_size, max_length, input_size = X.shape
        
        # Transpose to (batch, channels, time) format for convolution
        # X is (batch, time, features) -> (batch, features, time)
        X_t = np.transpose(X, (0, 2, 1))  # (batch_size, input_size, max_length)
        
        # Apply parallel convolutions with different kernel sizes
        conv_outputs = []
        conv_caches = []
        
        for conv_layer in self.conv_layers:
            # Conv output: (batch, num_filters, output_length)
            conv_out = conv_layer.forward(X_t)
            conv_out = relu(conv_out)
            conv_outputs.append(conv_out)
            conv_caches.append(conv_out)
        
        # Global max pooling over time dimension for each filter
        pooled_outputs = []
        for conv_out in conv_outputs:
            # Global max pooling: max over time dimension
            # conv_out: (batch, num_filters, time)
            pooled = np.max(conv_out, axis=2)  # (batch, num_filters)
            pooled_outputs.append(pooled)
        
        # Concatenate outputs from all kernel sizes
        # Each pooled is (batch, num_filters), concatenate along feature dimension
        concatenated = np.concatenate(pooled_outputs, axis=1)  # (batch, num_kernels * num_filters)
        
        # Fully connected layers
        fc1_out = self.fc1.forward(concatenated)
        fc1_out = relu(fc1_out)
        
        logits = self.fc2.forward(fc1_out)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        cache = {
            'X_t': X_t,
            'conv_outputs': conv_outputs,
            'conv_caches': conv_caches,
            'pooled_outputs': pooled_outputs,
            'concatenated': concatenated,
            'fc1_out': fc1_out,
            'logits': logits
        }
        
        return probs, cache
    
    def backward(self, probs, y_onehot, cache):
        """
        Backward pass through 1D CNN.
        
        Args:
            probs: Predicted probabilities (batch_size, num_classes)
            y_onehot: One-hot encoded true labels (batch_size, num_classes)
            cache: Dictionary from forward pass
        
        Returns:
            grads: Dictionary of gradients (not used, but kept for consistency)
        """
        batch_size = probs.shape[0]
        
        # Gradient of cross-entropy loss w.r.t. logits: dL/dlogits = probs - y_onehot
        dlogits = probs - y_onehot
        
        # Backward through fully connected layers
        dfc1_out = self.fc2.backward(dlogits)
        dfc1_out = relu_backward(dfc1_out, cache['fc1_out'])
        dconcatenated = self.fc1.backward(dfc1_out)
        
        # Split gradient back to each kernel size branch
        num_filters = self.num_filters
        dpooled = []
        start_idx = 0
        for _ in self.kernel_sizes:
            end_idx = start_idx + num_filters
            dpooled.append(dconcatenated[:, start_idx:end_idx])
            start_idx = end_idx
        
        # Backward through global max pooling and convolutions
        dconv_outputs = []
        for i, (dpool, conv_out) in enumerate(zip(dpooled, cache['conv_outputs'])):
            # Global max pooling backward: distribute gradient to max positions
            dconv = np.zeros_like(conv_out)  # (batch, num_filters, time)
            batch_size, num_filters, time_len = conv_out.shape
            
            for b in range(batch_size):
                for f in range(num_filters):
                    max_idx = np.argmax(conv_out[b, f, :])
                    dconv[b, f, max_idx] = dpool[b, f]
            
            # Backward through ReLU
            dconv = relu_backward(dconv, cache['conv_caches'][i])
            
            # Backward through convolution
            dx_conv = self.conv_layers[i].backward(dconv)
            dconv_outputs.append(dx_conv)
        
        # Sum gradients from all branches (they all came from the same input)
        dX_t = sum(dconv_outputs)
        
        # Transpose back to (batch, time, features)
        dX = np.transpose(dX_t, (0, 2, 1))
        
        return {}
    
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
    
    def update_params(self, learning_rate, clip_value=5.0):
        """
        Update parameters using gradients with gradient clipping.
        
        Args:
            learning_rate: Learning rate
            clip_value: Maximum gradient norm for clipping
        """
        # Collect all gradients
        all_grads = []
        for conv in self.conv_layers:
            all_grads.append(conv.dw.flatten())
            all_grads.append(conv.db.flatten())
        all_grads.append(self.fc1.dw.flatten())
        all_grads.append(self.fc1.db.flatten())
        all_grads.append(self.fc2.dw.flatten())
        all_grads.append(self.fc2.db.flatten())
        
        total_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
        
        if total_norm > clip_value:
            clip_coef = clip_value / (total_norm + 1e-6)
            for conv in self.conv_layers:
                conv.dw *= clip_coef
                conv.db *= clip_coef
            self.fc1.dw *= clip_coef
            self.fc1.db *= clip_coef
            self.fc2.dw *= clip_coef
            self.fc2.db *= clip_coef
        
        # Update parameters
        for conv in self.conv_layers:
            conv.update(learning_rate)
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)


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


def train_cnn1d(model, X_train, y_train, X_dev, y_dev, sequence_lengths_train, 
                sequence_lengths_dev, learning_rate=0.01, num_epochs=50, 
                batch_size=16, verbose=True):
    """
    Train 1D CNN model.
    
    Args:
        model: CNN1DClassifier instance
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
            model.backward(probs, y_onehot, cache)
            
            # Update parameters
            model.update_params(learning_rate)
        
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
           title='Confusion Matrix - 1D CNN')
    
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
    Main function to train and evaluate 1D CNN model for tennis serve classification.
    Randomly splits 100 examples into train (70), dev (15), and test (15).
    """
    # Set random seed for reproducibility
    seed = 14
    np.random.seed(seed)
    
    # Paths
    project_dir = Path(__file__).parent.parent.parent
    keypoints_dir = project_dir / 'data' / 'keypoints'
    labels_path = project_dir / 'data' / 'mcp_data.csv'
    output_dir = project_dir / 'src' / 'cnn1d' / 'test_results'
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
    
    # Create and train 1D CNN model
    print("\n" + "="*50)
    print("Training 1D CNN Model")
    print("="*50)
    
    input_size = X_train_padded.shape[2]  # 75 keypoints
    num_classes = 3  # 4, 5, 6
    kernel_sizes = [3, 5, 7]  # Different kernel sizes to capture various temporal patterns
    
    model = CNN1DClassifier(
        input_size=input_size, 
        num_classes=num_classes,
        num_filters=32,
        kernel_sizes=kernel_sizes,
        hidden_size=64
    )
    
    print(f"Model configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Kernel sizes: {kernel_sizes}")
    print(f"  Filters per kernel: 32")
    print(f"  Hidden size: 64")
    print(f"  Number of classes: {num_classes}")
    print(f"  Max sequence length: {max_length}")
    
    # Train model
    train_losses, dev_losses, train_accuracies, dev_accuracies = train_cnn1d(
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
        test_pred_path = output_dir / f'cnn1d_test_predictions_seed_{seed}.txt'
        with open(test_pred_path, 'w') as f:
            f.write("Sample,True_Label,Predicted_Label,Confidence\n")
            for i in range(len(y_test)):
                true_label = direction_classes[y_test[i]]
                pred_label = direction_classes[test_preds[i]]
                confidence = test_probs[i, test_preds[i]]
                f.write(f"{i},{true_label},{pred_label},{confidence:.4f}\n")
        print(f"\nTest predictions saved to {test_pred_path}")
        
        # Plot confusion matrix
        cm_path = output_dir / f'cnn1d_confusion_matrix_seed_{seed}.png'
        plot_confusion_matrix(y_test, test_preds, direction_classes, str(cm_path))
        
        # Save accuracy summary
        summary_path = output_dir / f'cnn1d_accuracy_summary_seed_{seed}.txt'
        with open(summary_path, 'w') as f:
            f.write("1D CNN Model Results\n")
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


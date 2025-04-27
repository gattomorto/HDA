import matplotlib.pyplot as plt
import psutil
import tensorflow as tf
import numpy as np

import v4


# In utils.py, modify plot_momentum_histograms() like this:
def plot_abs_momentum_histograms(momenta, model):
    num_layers = len(momenta)
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))

    for i, ax in enumerate(axes):
        # Convert TensorFlow variable to NumPy array
        momenta_data = np.abs(momenta[i].numpy())  # <-- This is the key fix
        ax.hist(momenta_data, bins=50, color='steelblue', edgecolor='black')
        ax.set_title(f'Layer {i} Momentum Distribution')
        ax.set_xlabel('Momentum Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def mem_usage():
    mem_info = psutil.Process().memory_full_info()  # Retrieve memory full info
    # Return the tuple with all metrics except for `num_page_faults`
    return (
        tf.config.experimental.get_memory_info('CPU:0')['peak'] / (1024 * 1024),
        #mem_info.rss / (1024 * 1024),  # Resident Set Size (in MB)
        mem_info.vms / (1024 * 1024),  # Virtual Memory Size (in MB)
        mem_info.peak_wset / (1024 * 1024),  # Peak Working Set (in MB)
        #mem_info.wset / (1024 * 1024),  # Current Working Set (in MB)
        #mem_info.peak_paged_pool / (1024 * 1024),  # Peak Paged Pool (in MB)
        #mem_info.paged_pool / (1024 * 1024),  # Current Paged Pool (in MB)
        #mem_info.peak_nonpaged_pool / (1024 * 1024),  # Peak Non-Paged Pool (in MB)
        #mem_info.nonpaged_pool / (1024 * 1024),  # Current Non-Paged Pool (in MB)
        #mem_info.pagefile / (1024 * 1024),  # Pagefile Commit (in MB)
        mem_info.peak_pagefile / (1024 * 1024)  # Peak Pagefile Commit (in MB)
        #mem_info.private / (1024 * 1024),  # Private Memory (in MB)
        #mem_info.uss / (1024 * 1024)  # Unique Set Size (in MB)
    )


def generate_data(samples, features, classes, noise_std=0.1):
    # Randomly generate class weight vectors (each class gets one)
    W = np.random.randn(features, classes)
    X = np.random.randn(samples, features).astype(np.float32)
    logits = X @ W + noise_std * np.random.randn(samples, classes)
    labels = np.argmax(logits, axis=1)
    Y = tf.keras.utils.to_categorical(labels, num_classes=classes)
    return X, Y


def load_mnist_data(flatten=False):
    add_channel = True
    if flatten:
        add_channel = False

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    if flatten:
        # Flatten the images to [batch_size, 784]
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
    elif add_channel:
        # Add a channel dimension to get shape [batch_size, 28, 28, 1]
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    # else: keep shape [batch_size, 28, 28] (no flattening, no channel added)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)



import matplotlib.pyplot as plt
import numpy as np

def plot_weight_histogram_all(model,it):
    for i in range(model.num_layers):
        plot_weight_histogram(model,i,it)

def plot_weight_histogram(model, layer_idx, it):
    """
    Plot histogram of absolute weight values for a specific layer.

    Args:
        model: Your sparse FFN model (FFNsSparse3)
        layer_idx: Index of layer to visualize
        bins: Number of bins for histogram
        figsize: Size of the figure
    """
    if layer_idx < 0 or layer_idx >= model.num_layers:
        raise ValueError(f"Layer index must be between 0 and {model.num_layers - 1}")

    # Get the weight values for the specified layer
    weights = model.W_values[layer_idx].numpy()
    abs_weights = weights
    #abs_weights = np.abs(weights)

    plt.figure(figsize=(10, 6))
    plt.hist(abs_weights, bins=50, edgecolor='black')

    # Add vertical line at mean and median
    mean_val = np.mean(abs_weights)
    median_val = np.median(abs_weights)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.4f}')

    plt.title(f'Layer {layer_idx} Weight Magnitude Distribution\n'
              f'it: {it}, '
              #f'Non-zero weights: {len(weights)}/{np.prod(model.W_shapes[layer_idx])} '
              #f'({len(weights) / np.prod(model.W_shapes[layer_idx]):.2%})'
              )
    plt.xlabel('Absolute Weight Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print some statistics
    '''print(f"Layer {layer_idx} Weight Statistics:")
    print(f"  Min: {np.min(abs_weights):.6f}")
    print(f"  Max: {np.max(abs_weights):.6f}")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Median: {median_val:.6f}")
    print(f"  Std Dev: {np.std(abs_weights):.6f}")'''


def print_layer_sparsity(model):
    """Prints the sparsity percentage for each layer in the model"""
    print("\nLayer Sparsity:")
    for i in range(model.num_layers):
        nnz = v4.get_num_nonzero_weights(i, model)
        total = model.W_shapes[i][0] * model.W_shapes[i][1]
        sparsity = 100.0 * (1.0 - nnz/total)
        print(f"Layer {i}: {sparsity:.2f}% sparse ({nnz}/{total} non-zero weights)")
    total_nnz = v4.get_total_nonzero_weights(model)
    total_weights = sum(shape[0]*shape[1] for shape in model.W_shapes)
    overall_sparsity = 100.0 * (1.0 - total_nnz/total_weights)
    print(f"Overall: {overall_sparsity:.2f}% sparse ({total_nnz}/{total_weights} non-zero weights)")
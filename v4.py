import os
import sys
from guppy import hpy
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import tensorflow as tf
from cnn_utils import *
from memory_profiler import profile
from memory_profiler import memory_usage

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

def pprint_sparse_tensor(st):
  s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
  for (index, value) in zip(st.indices, st.values):
    s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
  return s + "}>"

def random_sparse_indices(shape, sparsity, seed=None):
    import numpy as np

    rows, cols = shape
    total = rows * cols
    nnz = int((1.0 - sparsity) * total)  # Number of non-zero elements

    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=nnz, replace=False)

    # Convert flat indices to 2D indices
    indices = np.stack(np.unravel_index(chosen, shape), axis=1)
    return tf.constant(indices, dtype=tf.int64),nnz

def random_sparse_indices2(shape, sparsity, seed=None):
    rows, cols = shape
    total = rows * cols
    nnz = int((1.0 - sparsity) * total)  # Number of non-zero elements

    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=nnz, replace=False)

    # Convert flat indices to 2D indices
    indices = np.stack(np.unravel_index(chosen, shape), axis=1)

    # Sort indices in row-major order (first by row, then by column)
    sorted_order = np.lexsort((indices[:, 1], indices[:, 0]))
    sorted_indices = indices[sorted_order]

    return tf.constant(sorted_indices, dtype=tf.int64), nnz

class FFNsSparse(tf.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, sparsity=0.0):

        super().__init__(name=None)

        self.sparsity = sparsity
        self.num_layers = num_hidden_layers + 1  # hidden layers + output layer

        # Dimensions for each layer
        layer_dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]

        self.W_indices = []
        self.W_values = []
        self.W_shapes = []
        self.b = []

        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            shape = [out_dim, in_dim]

            indices, nnz = random_sparse_indices(shape, sparsity=sparsity, seed=42 + i)
            values = tf.Variable(tf.random.normal([nnz]), name=f"W{i+1}_values")
            bias = tf.Variable(tf.zeros([out_dim, 1]), name=f"b{i+1}")

            self.W_indices.append(indices)
            self.W_values.append(values)
            self.W_shapes.append(shape)
            self.b.append(bias)


    def __call__(self, X):
        out = X
        for i in range(self.num_layers):
            W = tf.sparse.SparseTensor(indices=self.W_indices[i], values=self.W_values[i], dense_shape=self.W_shapes[i])
            out = tf.sparse.sparse_dense_matmul(W, out)

            #Wd = tf.sparse.to_dense(W)
            #out = tf.matmul(Wd, out)

            out = out + self.b[i]
            if i < self.num_layers - 1:
                out = tf.nn.relu(out)

        return out

# è come FFNsSparse, tranne che si aspetta X (batch_size, input_dim)
class FFNsSparse3(tf.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, sparsity=0.0):

        super().__init__(name=None)

        self.sparsity = sparsity
        self.num_layers = num_hidden_layers + 1  # hidden layers + output layer

        # Dimensions for each layer
        layer_dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]

        self.W_indices = [None] * self.num_layers
        self.W_values = [None] * self.num_layers
        self.W_shapes = [None] * self.num_layers
        self.b = [None] * self.num_layers

        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            shape = [in_dim, out_dim]

            #values = tf.Variable(tf.random.normal([nnz]), name=f"W{i+1}_values")
            #bias = tf.Variable(tf.zeros([out_dim, 1]), name=f"b{i+1}")

            self.W_indices[i], nnz = random_sparse_indices2(shape, sparsity=sparsity, seed=i)
            self.W_values[i] = tf.Variable(tf.random.normal([nnz]), name=f"W{i + 1}_values")
            self.W_shapes[i] = shape
            self.b[i] = tf.Variable(tf.zeros([1,out_dim]), name=f"b{i + 1}")


    def __call__(self, X):
        out = X
        for i in range(self.num_layers):
            W = tf.sparse.SparseTensor(indices=self.W_indices[i], values=self.W_values[i], dense_shape=self.W_shapes[i])
            out = tf.sparse.sparse_dense_matmul(out,W)

            #Wd = tf.sparse.to_dense(W)
            #out = tf.matmul(Wd, out)

            out = out + self.b[i]
            if i < self.num_layers - 1:
                out = tf.nn.relu(out)

        return out


class DenseFFN(tf.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        """
        input_dim: int
        hidden_dim: int (used for all hidden layers)
        output_dim: int
        num_hidden_layers: int (number of hidden layers)
        """
        super().__init__(name=None)

        self.num_layers = num_hidden_layers + 1  # hidden layers + output layer

        # Dimensions for each layer
        layer_dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]

        self.W = []
        self.b = []

        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            W = tf.Variable(tf.random.normal([out_dim, in_dim]), name=f"W{i+1}")
            b = tf.Variable(tf.zeros([out_dim, 1]), name=f"b{i+1}")

            self.W.append(W)
            self.b.append(b)

    def __call__(self, X):
        out = X
        for i in range(self.num_layers):
            out = tf.matmul(self.W[i], out) + self.b[i]
            if i < self.num_layers - 1:
                out = tf.nn.relu(out)
        return out

def prune_layer(i,momenta,model):
    idx = tf.argsort(tf.math.abs(momenta[i]))
    momentum_sorted = tf.gather(momenta[i], idx)
    indices_sorted = tf.gather(model.W_indices[i], idx)
    values_sorted = tf.gather(model.W_values[i], idx)

    split_idx = tf.shape(momentum_sorted)[0] // 2

    indices_new = indices_sorted[split_idx:]
    values_new = values_sorted[split_idx:]
    momentum_new = momentum_sorted[split_idx:]

    model.W_indices[i]=indices_new
    #model.W_values[i]=values_new
    #   model.W_values[i].assign(values_new)
    model.W_values[i] = tf.Variable(values_new, name=f"W{i + 1}_values", trainable=True)


def get_num_non_zero_weights(i,model):
    return tf.shape(model.W_indices[i])[0]




    x=0


    '''momentum_sorted = momentum[idx]
    indices_sorted = indices[idx]'''

def get_contribution(i,model,momenta):
    mean_momentum_contributions = []
    for layer_i in range(model.num_layers):
        mean_momentum = np.mean(np.abs(momenta[layer_i]))
        mean_momentum_contributions.append(mean_momentum)

    total_momentum = sum(mean_momentum_contributions)

    momentum_contribution_proportions = []
    for mean_momentum in mean_momentum_contributions:
        proportion = mean_momentum / total_momentum
        momentum_contribution_proportions.append(proportion)

    return momentum_contribution_proportions[i]



def train(model, X, Y, epochs, batch_size,lr ):
    start_time = time.time()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    momenta = []
    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        for step, (x_batch, y_batch) in enumerate(dataset):
            #x_batch = tf.transpose(x_batch, perm=[1, 0])

            with tf.GradientTape() as tape:
                logits = model(x_batch)
                #logits = tf.transpose(logits, perm=[1, 0])
                loss = loss_fn(y_batch, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 1 == 0:
                print(f"Step {step:03d} | Loss: {loss:.4f}")
                #print(mem_usage())
                #print("--- %s seconds ---" % (time.time() - start_time))

                #exit(-1)
                #print("W1_values:", model.W1_values.numpy())




            for var in optimizer.variables:
                if 'W' in var.path and 'momentum' in var.path:
                    momenta.append(var.value)

            print(get_num_non_zero_weights(0,model))

            prune_layer(0,momenta,model)
            print(get_num_non_zero_weights(0,model))
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            momenta=[]


def main():
    num_features = 100
    num_classes = 10
    hidden_dim =  40
    num_hidden_layers = 5




    X, Y = generate_data(samples = 100, features=num_features, classes=num_classes)

    model = FFNsSparse3(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes,
                        num_hidden_layers=num_hidden_layers, sparsity=0.511)
    #model = DenseFFN(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes, num_hidden_layers=num_hidden_layers)

    t = model.trainable_variables
    train(model, X, Y,epochs=5, batch_size=10,lr= 0.001)


if __name__ == '__main__':
    main()

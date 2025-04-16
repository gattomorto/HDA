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
        mem_info.rss / (1024 * 1024),  # Resident Set Size (in MB)
        mem_info.vms / (1024 * 1024),  # Virtual Memory Size (in MB)
        mem_info.peak_wset / (1024 * 1024),  # Peak Working Set (in MB)
        mem_info.wset / (1024 * 1024),  # Current Working Set (in MB)
        mem_info.peak_paged_pool / (1024 * 1024),  # Peak Paged Pool (in MB)
        mem_info.paged_pool / (1024 * 1024),  # Current Paged Pool (in MB)
        mem_info.peak_nonpaged_pool / (1024 * 1024),  # Peak Non-Paged Pool (in MB)
        mem_info.nonpaged_pool / (1024 * 1024),  # Current Non-Paged Pool (in MB)
        mem_info.pagefile / (1024 * 1024),  # Pagefile Commit (in MB)
        mem_info.peak_pagefile / (1024 * 1024),  # Peak Pagefile Commit (in MB)
        mem_info.private / (1024 * 1024),  # Private Memory (in MB)
        mem_info.uss / (1024 * 1024)  # Unique Set Size (in MB)
    )


def generate_data(samples=1000, features=100, classes=10):
    X = np.random.randn(samples, features).astype(np.float32)
    Y = tf.keras.utils.to_categorical(np.random.randint(0, classes, samples), num_classes=classes)
    return X, Y



class DeepDenseModel(tf.keras.Model):
    def __init__(self, units=512, depth=200, num_classes=10):
        super().__init__()
        self.hidden_layers = [tf.keras.layers.Dense(units) for _ in range(depth)]
        self.out_layer = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)


class DeepDenseCheckpointed(tf.keras.Model):
    def __init__(self, units=512, depth=200, num_classes=10):
        super().__init__()
        # self.hidden_layers è solo una lista
        self.hidden_layers = [tf.recompute_grad(tf.keras.layers.Dense(units)) for _ in range(depth)]
        #self.hidden_layers = [tf.keras.layers.Dense(units) for _ in range(depth)]
        self.out_layer = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        for layer in self.hidden_layers:
            #x = tf.recompute_grad(layer)(x) #The wrapper isn't preserved between calls
            x = layer(x)
        return self.out_layer(x)


def train(model, X, Y, epochs=10, batch_size=32):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 1 == 0:
                print(f"Step {step:03d} | Loss: {loss:.4f}")
                print(mem_usage())

def main():
    X, Y = generate_data(50, features=100)

    model = DeepDenseModel()
    #model = DeepDenseCheckpointed()
    train(model, X, Y)


if __name__ == '__main__':
    main()




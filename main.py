import os
import sys
from guppy import hpy

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

def compute_cost(outp, Y):
    cost = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred= outp, y_true = Y, from_logits=True))
    return cost

def train_model(network_model, X_train, Y_train, X_test, Y_test,
                learning_rate = 0.005, num_epochs = 300, minibatch_size = 128,
                print_cost = True):


    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    rss_values = []
    vms_values = []
    peak_wset_values = []
    wset_values = []
    peak_paged_pool_values = []
    paged_pool_values = []
    peak_nonpaged_pool_values = []
    nonpaged_pool_values = []
    pagefile_values = []
    peak_pagefile_values = []
    private_values = []
    uss_values = []

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Do the training loop
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed += 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            with tf.GradientTape() as tape:
                (minibatch_X, minibatch_Y) = minibatch
                outp = network_model(minibatch_X, training=True)
                cost = compute_cost(outp, minibatch_Y)

            gradients = tape.gradient(cost, network_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, network_model.trainable_variables))
            minibatch_cost += cost / num_minibatches

            metrics = mem_usage()
            rss_values.append(metrics[0])
            vms_values.append(metrics[1])
            peak_wset_values.append(metrics[2])
            wset_values.append(metrics[3])
            peak_paged_pool_values.append(metrics[4])
            paged_pool_values.append(metrics[5])
            peak_nonpaged_pool_values.append(metrics[6])
            nonpaged_pool_values.append(metrics[7])
            pagefile_values.append(metrics[8])
            peak_pagefile_values.append(metrics[9])
            private_values.append(metrics[10])
            uss_values.append(metrics[11])

            print(f"  RSS: {metrics[0]:.2f} MB")
            print(f"  VMS: {metrics[1]:.2f} MB")
            print(f"  Peak Working Set: {metrics[2]:.2f} MB")
            print(f"  Working Set: {metrics[3]:.2f} MB")
            print(f"  Peak Paged Pool: {metrics[4]:.2f} MB")
            print(f"  Paged Pool: {metrics[5]:.2f} MB")
            print(f"  Peak Non-Paged Pool: {metrics[6]:.2f} MB")
            print(f"  Non-Paged Pool: {metrics[7]:.2f} MB")
            print(f"  Pagefile Commit: {metrics[8]:.2f} MB")
            print(f"  Peak Pagefile Commit: {metrics[9]:.2f} MB")
            print(f"  Private Memory: {metrics[10]:.2f} MB")
            print(f"  Unique Set Size: {metrics[11]:.2f} MB")
            print("-" * 50)





        '''# Print the cost every epoch
        if print_cost == True and epoch % 5 == 0:
            pass
            #print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)'''


    # Calculate accuracy on the validation set
    #train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_model(X_train, training=False), 1), tf.argmax(Y_train, 1)), "float"))
    #test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_model(X_test, training=False), 1), tf.argmax(Y_test, 1)), "float"))

    #print("Train Accuracy:", train_accuracy)
    #print("Test Accuracy:", test_accuracy)

    #plt.figure(figsize=(10, 6))

    # Plot each metric
    plt.plot(rss_values, label='RSS (Resident Set Size)', color='blue')
    plt.plot(vms_values, label='VMS (Virtual Memory Size)', color='green')
    plt.plot(peak_wset_values, label='Peak Working Set', color='red')
    plt.plot(wset_values, label='Working Set', color='purple')
    plt.plot(peak_paged_pool_values, label='Peak Paged Pool', color='orange')
    plt.plot(paged_pool_values, label='Paged Pool', color='cyan')
    plt.plot(peak_nonpaged_pool_values, label='Peak Non-Paged Pool', color='brown')
    plt.plot(nonpaged_pool_values, label='Non-Paged Pool', color='pink')
    plt.plot(pagefile_values, label='Pagefile Commit', color='grey')
    plt.plot(peak_pagefile_values, label='Peak Pagefile Commit', color='yellow')
    plt.plot(private_values, label='Private Memory', color='black')
    plt.plot(uss_values, label='Unique Set Size', color='magenta')
    plt.legend()
    plt.grid(True)
    #plt.tight_layout()
    plt.show()

    return network_model

if __name__ == '__main__':

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    print(f"Array size: {X_train_orig.nbytes/ (1024 * 1024):.2f} MB")
    print(f"Array size: {Y_train_orig.nbytes/ (1024 * 1024):.2f} MB")

    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    #X_train_cast = X_train.astype(dtype=np.float32)
    #X_test_cast = X_test.astype(dtype=np.float32)

    network_model = tf.keras.Sequential(
        [tf.keras.layers.Conv2D(4, (3, 3), strides=(1, 1), padding='same', activation=None),
         tf.keras.layers.BatchNormalization(axis=-1),
         tf.keras.layers.Activation('relu'),

         tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'),

         tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation=None),
         tf.keras.layers.BatchNormalization(axis=-1),
         tf.keras.layers.Activation('relu'),

         tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'),

         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(16, activation='relu'),
         tf.keras.layers.Dense(6, activation=None)])

    #train_model(network_model, X_train, Y_train, X_test, Y_test, minibatch_size = 64,num_epochs=5,learning_rate= 0.001)

    mem_usage = memory_usage((train_model,
                              (network_model, X_train, Y_train, X_test, Y_test),
                              {'learning_rate': 0.001, 'num_epochs': 5, 'minibatch_size': 64}))
    peak_mem = max(mem_usage)
    print(f"Peak memory usage: {peak_mem} MiB")
    plt.figure(figsize=(10, 6))
    plt.plot(mem_usage)
    plt.show()



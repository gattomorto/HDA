import gc
import os
import sys
import time
#from guppy import hpy
import numpy as np
#import matplotlib.pyplot as plt
import psutil
import tensorflow as tf
from cnn_utils import *
#from memory_profiler import profile
#from memory_profiler import memory_usage
from tensorflow.keras import mixed_precision



red = False
tf.keras.backend.set_floatx('float32')
mixed_precision.set_global_policy("float32")
#mixed_precision.set_global_policy("mixed_bfloat16")
print("Compute dtype:", mixed_precision.global_policy().compute_dtype)  # Should print 'bfloat16'
print("Variable dtype:", mixed_precision.global_policy().variable_dtype)  # Should print 'float32'
print(tf.keras.backend.floatx())

gc.collect()
def mem_usage(printt = False):
    mem_info = psutil.Process().memory_full_info()  # Retrieve memory full info
    # Return the tuple with all metrics except for `num_page_faults`
    mem= (
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
    if printt:
        print(f"  RSS: {mem[0]:.2f} MB")
        print(f"  VMS: {mem[1]:.2f} MB")
        print(f"  Peak Working Set: {mem[2]:.2f} MB")
        print(f"  Working Set: {mem[3]:.2f} MB")
        print(f"  Peak Paged Pool: {mem[4]:.2f} MB")
        print(f"  Paged Pool: {mem[5]:.2f} MB")
        print(f"  Peak Non-Paged Pool: {mem[6]:.2f} MB")
        print(f"  Non-Paged Pool: {mem[7]:.2f} MB")
        print(f"  Pagefile Commit: {mem[8]:.2f} MB")
        print(f"  Peak Pagefile Commit: {mem[9]:.2f} MB")
        print(f"  Private Memory: {mem[10]:.2f} MB")
        print(f"  Unique Set Size: {mem[11]:.2f} MB")

    return mem

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
    #optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(), dynamic=True)

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

            metrics = mem_usage(printt=True)

            gpu_memory = tf.config.experimental.get_memory_info('CPU:0')
            print(gpu_memory)
            exit()
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



        # Print the cost every epoch
        if print_cost == True and epoch % 1 == 0:
            #pass
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)


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



class CustomCNN(tf.keras.Model):
    def __init__(self, input_shape, num_classes=6, num_conv_blocks=2, num_dense_layers=1, base_filters=4, grad_chek = False):
        super(CustomCNN, self).__init__()

        self.conv_blocks = []
        for i in range(num_conv_blocks):
            filters = base_filters * (2 ** i)
            if grad_chek:
                self.conv_blocks.append([
                    tf.recompute_grad(tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation=None)),
                    tf.recompute_grad(tf.keras.layers.BatchNormalization()),
                    tf.recompute_grad(tf.keras.layers.Activation('relu')),
                    tf.recompute_grad(tf.keras.layers.MaxPool2D((2, 2), padding='same'))
                ])
            else:
                if red:
                      self.conv_blocks.append([
                      tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation=None,dtype="float16"),
                      tf.keras.layers.BatchNormalization(dtype="float16"),
                      tf.keras.layers.Activation('relu',dtype="float16"),
                      tf.keras.layers.MaxPool2D((2, 2), padding='same',dtype="float16")
                  ])
                else:
                      self.conv_blocks.append([
                      tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation=None),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.Activation('relu'),
                      tf.keras.layers.MaxPool2D((2, 2), padding='same')
                  ])

        self.flatten = tf.keras.layers.Flatten()

        if grad_chek:
            self.dense_layers = [tf.recompute_grad(tf.keras.layers.Dense(64, activation='relu')) for _ in range(num_dense_layers)]
            #self.output_layer = tf.recompute_grad(tf.keras.layers.Dense(num_classes, activation=None))

        else:
            if red:
              self.dense_layers = [tf.keras.layers.Dense(64, activation='relu',dtype = 'float16') for _ in range(num_dense_layers)]
            else:
               self.dense_layers = [tf.keras.layers.Dense(64, activation='relu') for _ in range(num_dense_layers)]



        self.output_layer = tf.keras.layers.Dense(num_classes, activation=None)

    def call(self, inputs, training=False):
        x = inputs
        for block in self.conv_blocks:
            for layer in block:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    x = layer(x, training=training)  # Important: pass training flag
                else:
                    x = layer(x)

        x = self.flatten(x)

        for dense in self.dense_layers:
            x = dense(x)

        return self.output_layer(x)


def main():
    devices = tf.config.list_physical_devices()
    print("All devices:")
    for device in devices:
        print(device)

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


    input_shape = X_train.shape[1:]
    network_model = CustomCNN(input_shape, num_classes=6, num_conv_blocks=10, num_dense_layers=10,grad_chek=False)
    #network_model.build(input_shape=(None, *input_shape))  # Needed to initialize weights

    for layer in network_model.layers:
      print(f"Layer: {layer.name}, Dtype: {layer.dtype}")

    #start_time = time.time()
    train_model(network_model, X_train, Y_train, X_test, Y_test, minibatch_size = 256,num_epochs=5,learning_rate= 0.001)
    #print("--- %s seconds ---" % (time.time() - start_time))



main()
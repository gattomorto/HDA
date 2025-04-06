import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_utils import *
from memory_profiler import profile
from memory_profiler import memory_usage



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

        # Print the cost every epoch
        if print_cost == True and epoch % 5 == 0:
            pass
            #print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)


    # Calculate accuracy on the validation set
    #train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_model(X_train, training=False), 1), tf.argmax(Y_train, 1)), "float"))
    #test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_model(X_test, training=False), 1), tf.argmax(Y_test, 1)), "float"))

    #print("Train Accuracy:", train_accuracy)
    #print("Test Accuracy:", test_accuracy)

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
    plt.title('Memory Usage During Model Training')
    plt.xlabel('Time (intervals of 0.1s)')
    plt.ylabel('Memory Usage (MiB)')
    plt.grid(True)
    plt.savefig('memory_usage_plot.png')  # Save the plot to a file
    plt.show()



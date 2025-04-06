import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_utils import *
from memory_profiler import profile


def compute_cost(outp, Y):
    """
    Computes the cost

    Arguments:
    outp -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as outp

    Returns:
    cost - Tensor of the cost function
    """

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred= outp, y_true = Y, from_logits=True))
    ### END CODE HERE ###

    return cost

@profile
def train_model(network_model, X_train, Y_train, X_test, Y_test, learning_rate = 0.005,
          num_epochs = 300, minibatch_size = 128, print_cost = True):
    """
    Train a ConvNet in TensorFlow

    Arguments:
    network_model -- the keras Sequential model to be trained
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- training set, of shape (None, n_y = 6)
    X_test -- test set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    validation_accuracy -- real number, testing accuracy on the validation set (X_val)
    """

    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    ### END CODE HERE ###

    # Do the training loop
    for epoch in range(num_epochs):

        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed += 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            with tf.GradientTape() as tape:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                ### START CODE HERE ### (1 line)
                outp = network_model(minibatch_X, training=True)
                ### END CODE HERE ###

                # Cost function
                ### START CODE HERE ### (1 line)
                cost = compute_cost(outp, minibatch_Y)
                ### END CODE HERE ###

            # Compute the gradient
            ### START CODE HERE ### (1 line)
            gradients = tape.gradient(cost, network_model.trainable_variables)
            ### END CODE HERE ###

            # Apply the optimizer
            ### START CODE HERE ### (1 line)
            optimizer.apply_gradients(zip(gradients, network_model.trainable_variables))
            ### END CODE HERE ###

            minibatch_cost += cost / num_minibatches

        # Print the cost every epoch
        if print_cost == True and epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Calculate accuracy on the validation set
    train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_model(X_train, training=False), 1), tf.argmax(Y_train, 1)), "float"))
    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(network_model(X_test, training=False), 1), tf.argmax(Y_test, 1)), "float"))

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    return train_accuracy, test_accuracy, network_model

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

    _, _, network_model_trained = train_model(network_model, X_train, Y_train, X_test, Y_test, minibatch_size = 1080,num_epochs=5,learning_rate= 0.001)




import tensorflow as tf
import numpy as np
import random


SEED = 0
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def load_bloodmnist_224():
    data = np.load("bloodmnist_224.npz")
    X_train = data['train_images']  # shape: (N, 224, 224, 3)
    y_train = data['train_labels']  # shape: (N,)
    X_test =  data['test_images']
    y_test =  data['test_labels']

    print(X_train.shape)

    # Normalize inputs and one-hot encode labels
    X_train = X_train.astype("float32") / 255.0

    print("0")
    X_test = X_test.astype("float32") / 255.0
    print("0.1")

    y_train = to_categorical(y_train)
    print("0.2")

    y_test = to_categorical(y_test)
    print("0.3")


    return (X_train, y_train), (X_test, y_test)

def load_bloodmnist_224_new(save_subset=True):
    data = np.load("bloodmnist_224.npz")
    X_train = data['train_images']
    y_train = data['train_labels']
    X_test = data['test_images']
    y_test = data['test_labels']

    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Save 10% subset of training data
    if save_subset:
        num_subset = int(0.1 * len(X_train))
        indices = np.random.choice(len(X_train), num_subset, replace=False)
        X_small = X_train[indices]
        y_small = y_train[indices]
        np.savez("bloodmnist_subset.npz", X=X_small, y=y_small)
        print(f"Saved 10% subset to bloodmnist_subset.npz ({num_subset} samples)")

    return (X_train, y_train), (X_test, y_test)

def load_bloodmnist_subset():
    data = np.load("bloodmnist_subset.npz")
    X = data['X']
    y = data['y']
    print(f"Loaded subset: {X.shape}, {y.shape}")
    return X, y

def test(model, X, y):
    print(X.shape)
    logits = model(X[:8])

    preds = tf.argmax(logits, axis=1)

    true_labels = tf.argmax(y[:8], axis=1)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, true_labels), tf.float32))

    return accuracy.numpy()

def test2(model, X, y, batch_size=32):
    num_samples = X.shape[0]
    num_correct = 0

    for i in range(0, num_samples, batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        logits = model(x_batch, training=False)
        preds = tf.argmax(logits, axis=1)
        true_labels = tf.argmax(y_batch, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, true_labels), tf.float32))
        num_correct += correct.numpy()

    accuracy = num_correct / num_samples
    return accuracy



def train(model, X, y, epochs, batch_size, lr, prune_and_regrow_step):

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(1024).batch(batch_size)

    it = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        for step, (x_batch, y_batch) in enumerate(dataset):
            it += 1
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)

                loss = loss_fn(y_batch, logits)
                print(f"loss: {loss.numpy():.3f}")

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if it % 1 == 0:
                pass
                #acc = test2(model, X, y)
                #print(f"it: {it}, acc: {acc:.3f}")


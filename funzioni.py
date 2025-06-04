import tensorflow as tf
import numpy as np
import random
import utils
import gc
from tensorflow.keras.utils import to_categorical
#from google.colab import drive
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output  # Works in Jupyter; ignored in scripts

'''if not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')
folder_path = '/content/drive/MyDrive/hda'
import sys'''

SEED = 0
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def load_bloodmnist_224():
    data = np.load("bloodmnist_224.npz")
    #data = np.load(f"{folder_path}/bloodmnist_224.npz")

    X_train = data['train_images'].astype("float32") / 255.0
    y_train = data['train_labels']
    X_test = data['test_images'].astype("float32") / 255.0
    y_test = data['test_labels']
    X_val = data['val_images'].astype("float32") / 255.0
    y_val = data['val_labels']


    # Always shuffle training data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

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
    #data = np.load(f"{folder_path}/bloodmnist_224.npz")

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

# versione con checkpoint
'''
def train(model, X, y, epochs, batch_size, lr, prune_and_regrow_step):
    SEED = 0
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    #dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(1024).batch(batch_size)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

    # === Checkpoint Setup ===
    checkpoint_dir = './checkpoints'
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)
    # Restore latest checkpoint if it exists
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Training from scratch")
    it = int(ckpt.step.numpy())

    #it = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")

        for _, (x_batch, y_batch) in enumerate(dataset):
            print(optimizer.learning_rate.numpy())
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                #print("dopo logits = model(x_batch, training=True)")
                #print(utils.mem_usage())
                loss = loss_fn(y_batch, logits)
                #print("dopo loss = loss_fn(y_batch, logits)")
                #print(utils.mem_usage())
                print(f"it: {it}, loss: {loss.numpy()}")

            grads = tape.gradient(loss, model.trainable_variables)
            #print("dopo grads = tape.gradient(loss, model.trainable_variables)")
            #print(utils.mem_usage())


            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            #print("dopo optimizer.apply_gradients(zip(grads, model.trainable_variables))")
            #print(utils.mem_usage())
            it = it + 1
            ckpt.step.assign(it)


            if it % 5 == 0:
                pass
                #acc = test2(model, X, y)
                #print(f"it: {it}, acc: {acc:.3f}")
                optimizer.learning_rate.assign(0.1)
                #print(utils.mem_usage())
                #save_path = manager.save()
                #print(f"Checkpoint saved at step {it}: {save_path}")
'''

'''
def train(model, X, y, epochs, batch_size, lr, prune_and_regrow_step, patience=2):
    SEED = 0
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

    # === Checkpoint Setup ===
    checkpoint_dir = './checkpoints'
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)
    # Restore latest checkpoint if it exists
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Training from scratch")
    it = int(ckpt.step.numpy())

    # Variables for learning rate scheduling
    best_loss = float('inf')
    patience_counter = 0
    epoch_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        epoch_loss = 0
        num_batches = 0

        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
                epoch_loss += loss.numpy()

            num_batches += 1
            print(f"it: {it}, loss: {loss.numpy()}")
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            it += 1
            ckpt.step.assign(it)

            if it % 5 == 0:
                acc = test2(model, X, y)
                print(f"it: {it}, acc: {acc:.3f}")
                save_path = manager.save()
                print(f"Checkpoint saved at step {it}: {save_path}")

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        # Learning rate adjustment logic
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")

            if patience_counter >= patience:
                new_lr = optimizer.learning_rate.numpy() * 0.5
                optimizer.learning_rate.assign(new_lr)
                print(f"Reducing learning rate to {new_lr:.6f}")
                patience_counter = 0  # Reset counter after LR adjustment
'''




def train(
    model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    epochs,
    max_iter,
    batch_size,
    lr,
    prune_and_regrow_stride,
    patience=3,
    plot_every=1,
    live_plotting=True,
    weights_chekpoint_stride = 3,
    rho0=0.5):

    # Set random seeds for reproducibility
    SEED = 0
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(batch_size)

    checkpoint_dir = './checkpoints'
    #checkpoint_dir = '/content/drive/MyDrive/hda/checkpoints'
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Training from scratch")
    it = int(ckpt.step.numpy())

    # Tracking variables
    best_loss = float('inf')
    patience_counter = 0
    step_losses = []  # Loss at every step
    step_numbers = []  # Corresponding step numbers

    # Initialize plot (if live plotting)
    if live_plotting:
        plt.figure(figsize=(10, 6))
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Live Training Loss')
        plt.grid(True)

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        epoch_loss = 0
        num_batches = 0

        for step, (x_batch, y_batch) in enumerate(dataset):
            # Forward + backward pass
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
                epoch_loss += loss.numpy()

            # Store loss and step
            step_losses.append(loss.numpy())
            step_numbers.append(it)
            num_batches += 1

            # Optimization
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            it += 1
            if it == max_iter:
                print("max iter reached")
                return

            ckpt.step.assign(it)

            # Print progress
            print(f"Step {it}, Loss: {loss.numpy()}")
            #print(utils.mem_usage())

            # Live plotting
            if live_plotting and (it % plot_every == 0):
                try:
                    clear_output(wait=True)  # Works in Jupyter
                except:
                    plt.clf()  # Fallback for scripts
                plt.plot(step_numbers, step_losses, 'b-', label='Training Loss')
                plt.legend()
                plt.draw()
                plt.pause(0.01)  # Required for non-Jupyter environments

            # Checkpoint & validation
            if it % weights_chekpoint_stride == 0:
                acc_tr = test2(model, X_tr, y_tr)
                acc_val = test2(model, X_val, y_val)
                #acc_val = -1
                #acc_tr = -1
                print(f"Step {it}, Accuracy Train: {acc_tr:.3f},  Accuracy Val: {acc_val:.3f}")
                save_path = manager.save()
                print(f"Checkpoint saved: {save_path}")

            if it % prune_and_regrow_stride == 0:
                print("Prune & Regrow")
                #model.prune_and_regrow(0.5, optimizer)
                model.prune_and_regrow(rho0 ** (int(it/prune_and_regrow_stride)), optimizer)
                #TODO: inizializzare le nuove variabili con vecchi momenti?
                optimizer = tf.keras.optimizers.Adam(learning_rate=float(optimizer.learning_rate.numpy()))

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} Avg Loss: {avg_epoch_loss}")

        # Early learning rate adjustment
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                new_lr = optimizer.learning_rate.numpy() * 0.5
                optimizer.learning_rate.assign(new_lr)
                print(f"Reducing LR to {new_lr:.6f}")
                patience_counter = 0

        print(f"Patience counter: {patience_counter}")


        '''print("Prune & Regrow")
        # print(model)
        model.prune_and_regrow(rho0**epoch+1, optimizer)
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(optimizer.learning_rate.numpy()))'''

    # Final plot
    if live_plotting:
        clear_output(wait=True)
        plt.plot(step_numbers, step_losses, 'b-', label='Training Loss')
        plt.title('Final Training Loss')
        plt.legend()
        plt.show()






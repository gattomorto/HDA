import os
import sys
#from guppy import hpy
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.gen_sparse_ops import sparse_reorder

from cnn_utils import *
#from memory_profiler import profile
#from memory_profiler import memory_usage

from utils import *
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(0)
np.random.seed(0)


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


# è come FFNsSparse, tranne che si aspetta X (batch_size, input_dim)
#TODO: set indices, set values
#TODO: sono sicuro che indices deve essere tf.Constant anche se viene modificato in prune & regrow?
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
            self.W_values[i] = tf.Variable(tf.random.normal([nnz]), name=f"W{i}_values")
            self.W_shapes[i] = shape
            self.b[i] = tf.Variable(tf.zeros([1,out_dim]), name=f"b{i}")

    # out è denso, quindi cmq vai a caricare una matrice intera in memoria
    # TODO: studia grad_a, e grad_b di matmul
    def __call__(self, X):
        out = X
        for i in range(self.num_layers):
            W = tf.sparse.SparseTensor(indices=self.W_indices[i], values=self.W_values[i], dense_shape=self.W_shapes[i])

            #out = tf.sparse.sparse_dense_matmul(out,W)
            out = tf.matmul(out,tf.sparse.to_dense(W),b_is_sparse=True)

            out = out + self.b[i]
            if i < self.num_layers - 1:
                out = tf.nn.relu(out)

        return out


class DenseFFN(tf.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):

        super().__init__(name=None)

        self.num_layers = num_hidden_layers + 1  # hidden layers + output layer

        # Dimensions for each layer
        layer_dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]

        self.W = []
        self.b = []

        for i in range(self.num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            W = tf.Variable(tf.random.normal([in_dim, out_dim]), name=f"W{i}")
            b = tf.Variable(tf.zeros([1,out_dim]), name=f"b{i}")

            self.W.append(W)
            self.b.append(b)

    def __call__(self, X):
        out = X
        for i in range(self.num_layers):
            out = tf.matmul(out,self.W[i]) + self.b[i]
            if i < self.num_layers - 1:
                out = tf.nn.relu(out)
        return out


#TODO: serve fare tf.constant quando si creano nuovi indici?
def prune_layer(i, model):

    #plot_weight_histogram(model,i,50)
    idx = tf.argsort(tf.math.abs(model.W_values[i]))

    values_sorted =  tf.gather(model.W_values[i], idx)
    indices_sorted = tf.gather(model.W_indices[i], idx)


    split_idx = tf.shape(values_sorted)[0] // 30

    indices_new = indices_sorted[split_idx:]
    values_new = values_sorted[split_idx:]

    values_scartati = values_sorted[0:split_idx].numpy()
    abs_values_scartati = np.abs(values_scartati)
    mean_scartati = np.mean(abs_values_scartati)
    max_scartati = np.max(abs_values_scartati)
    print(f"Layer i: {i}, Num Scartati: {len(values_scartati)},  Media scartati: {mean_scartati:.3f}, Max scartati: {max_scartati:.3f}")



    model.W_indices[i] = indices_new
    model.W_values[i] = tf.Variable(values_new, name=f"W{i}_values", trainable=True)

    return split_idx.numpy()

def prune(model,verbose):
    tot_pruned = 0
    if verbose:
        print(f"Total Non-zero weights: {get_total_nonzero_weights(model)}")

    for i in range(model.num_layers):
        nz_before_i = get_num_nonzero_weights(i, model)
        pruned_i = prune_layer(i, model)
        nz_after_i = get_num_nonzero_weights(i, model)
        tot_pruned = tot_pruned + pruned_i
        if verbose:
            print(f"Layer: {i}, Non-zero weights before: {nz_before_i}, Pruned: {pruned_i}, Non-zero weights after: {nz_after_i}")
    if verbose:
        print(f"Tot pruned: {tot_pruned}")
        print(f"Total Non-zero weights: {get_total_nonzero_weights(model)}")

    return tot_pruned


#TODO: fanno schifo i nomi e get_contributions ha degli zeri per layers che non sono in non_saturated_layer -- fa schifo
#TODO: layers in teeoria non serve pechè li ottieni da m
#TODO: cambiare il tipo di eccezzione
#TODO: actual regrow diversio da effective pruned o qaulcosa
def regrow(model, momenta, to_regrow, layers,verbose=False):
    nnz_before = get_total_nonzero_weights(model)
    true_prop_contributions = get_contributions(layers, momenta)
    print(true_prop_contributions)
    tot_regrown = 0
    remaining = to_regrow

    while remaining != 0:
        prop_contributions = get_contributions(layers, momenta)
        tot_missing = 0
        non_saturated_layers = []
        for l in layers:
            expected_regrow = int(remaining * prop_contributions[l])
            actual_regrow = regrow_layer(l, model, expected_regrow)
            tot_regrown = tot_regrown + actual_regrow
            missing = expected_regrow - actual_regrow
            tot_missing = tot_missing + missing
            is_saturated = get_num_zero_weights(l,model)==0
            if not is_saturated:
                non_saturated_layers.append(l)
            if verbose:
                print(f"Layer: {l}, Regrown: {actual_regrow}, Saturated: {is_saturated}")

        layers = non_saturated_layers
        remaining = tot_missing
        if verbose & (tot_missing != 0):
            print(f"Some layers are saturated, missing {tot_missing}, redistribution...")

    nnz_after = get_total_nonzero_weights(model)

    test1 = 0
    test2 = 0
    if nnz_before + to_regrow != nnz_after:
        test1 = 1

    if to_regrow != tot_regrown:
        test2 = 1

    if(test1 != test2):
        raise Exception("i due test devono corrispondere")

    debdt = to_regrow-tot_regrown
    if debdt < 0:
        raise Exception("debdt strano")

    return debdt



def regrow_layer(i, model, desired_growth):

    shape = model.W_shapes[i]
    all_coords = np.array([(r, c) for r in range(shape[0]) for c in range(shape[1])], dtype=np.int64)

    existing = set(map(tuple, model.W_indices[i].numpy()))
    mask = np.array([tuple(coord) not in existing for coord in all_coords])
    available_coords = all_coords[mask]
    num_available_coords = len(available_coords)

    if desired_growth > num_available_coords:
        actual_regrow = num_available_coords
    else:
        actual_regrow = desired_growth

    chosen = available_coords[np.random.choice(len(available_coords), size=actual_regrow, replace=False)]

    new_indices = tf.constant(chosen, dtype=tf.int64)
    #new_values = tf.random.normal([actual_regrow])
    new_values = tf.zeros([actual_regrow])
    #new_values = tf.random.truncated_normal([actual_regrow],mean = 0, stddev = 0.01)

    model.W_indices[i] = tf.concat([model.W_indices[i], new_indices], axis=0)
    model.W_values[i] = tf.Variable(tf.concat([model.W_values[i], new_values], axis=0),
                                    name=f"W{i}_values", trainable=True)

    reorder_indices_and_values(model,i)

    return actual_regrow

#TODO: cosa fare con import? perchè non ce tf. davanti
def reorder_indices_and_values(model,i):
    indices = model.W_indices[i]
    values = model.W_values[i]
    from tensorflow.python.ops import gen_sparse_ops
    xx = sparse_reorder(indices,values,model.W_shapes[i])
    model.W_indices[i] = xx.output_indices
    model.W_values[i].assign(xx.output_values)


def prune_and_regrow(model,momenta,debdt):
    tot_pruned = prune( model, verbose=False)
    debdt = regrow(model, momenta, tot_pruned+debdt, [l for l in range(model.num_layers)])
    print("tot nonzero weights: ", get_total_nonzero_weights(model))
    return debdt


def get_total_nonzero_weights(model):
    tot = 0
    for i in range(model.num_layers):
        tot = tot + get_num_nonzero_weights(i,model)
    return tot

def get_num_nonzero_weights(i,model):
    return tf.shape(model.W_indices[i])[0].numpy()

def get_num_zero_weights(i,model):
    shape = model.W_shapes[i]
    return shape[0]*shape[1] - get_num_nonzero_weights(i,model)

#TODO: cambia il tipo di eccezione
#TODO: riscrivi in forma vettoriale?
#TODO: proprio non mi piace cosi che mean momentas ha gli 0 per i valori non validi
def get_contributions(layers, momenta):
    if layers==[]:
        raise Exception("layers empty, probably to_grow > available space")
    # ricorda che momenta è una lista di vettori
    mean_momentas = [0 for _ in range(len(momenta))] #forse è meglio sostituire con model.num_layers
    for l in layers:
        mean_momentum = np.mean(np.abs(momenta[l]))
        mean_momentas[l] = mean_momentum

    total_momentum = sum(mean_momentas)
    if total_momentum == 0:
        momentum_contribution = np.zeros(len(momenta))
        momentum_contribution[layers] = 1 / len(layers)
        return momentum_contribution

    momentum_contribution = []
    for mean_momentum in mean_momentas:
        proportion = mean_momentum / total_momentum
        momentum_contribution.append(proportion)

    return momentum_contribution


def train(model, X, y, epochs, batch_size,lr ,prune_and_regrow_step ):
    start_time = time.time()
    debdt=0
    it = 0
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        for step, (x_batch, y_batch) in enumerate(dataset):
            #x_batch = tf.transpose(x_batch, perm=[1, 0])
            it=it+1

            with tf.GradientTape() as tape:
                logits = model(x_batch)
                #logits = tf.transpose(logits, perm=[1, 0])
                loss = loss_fn(y_batch, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 1 == 0:
                acc = test(model,X,y)
                print(f"it: {it}, acc: {acc:.3f}")
                #print(f"Step {step:03d} | Loss: {loss:.4f}")
                #print(mem_usage())
                #print("--- %s seconds ---" % (time.time() - start_time))
                #exit(-1)
                #print("W1_values:", model.W1_values.numpy())

            if it % prune_and_regrow_step == 0:
                #plot_weight_histogram(model,model.num_layers-1)
                #plot_weight_histogram_all(model,f"before {it}")
                #TODO: W_momentum
                momenta = []
                velocities = []

                #https://github.com/keras-team/keras/blob/v3.3.3/keras/src/optimizers/base_optimizer.py#L567-L583
                print("prune & regrow")
                #beta1 = 0.9
                #beta2 = 0.999
                #momentum_correction = 1-beta1**prune_and_regrow_step
                #velocity_correction = 1-beta2**prune_and_regrow_step
                for var in optimizer.variables:
                    if 'W' in var.path and 'momentum' in var.path:
                        momenta.append(var.value)
                    #if 'W' in var.path and 'velocity' in var.path:
                    #    velocities.append(var.value)

                #for i in range(model.num_layers):
                #    momenta[i] = momenta[i]/(velocities[i] + optimizer.epsilon)


                debdt=prune_and_regrow(model,momenta,debdt)
                print_layer_sparsity(model)
                #plot_weight_histogram_all(model,f"after {it}")


                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                momenta = []



def test(model, X, y):
    logits = model(X)
    preds = tf.argmax(logits, axis=1)
    true_labels = tf.argmax(y, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, true_labels), tf.float32))
    return accuracy.numpy()


#TODO: capire. perchè hidden_dim = 1 dà problemi
def main():
    num_features = 784
    num_classes = 10
    hidden_dim = 256
    num_hidden_layers = 5
    (X_train,y_train),(X_test,y_test)= load_mnist_data()

    #X_train, y_train = generate_data(samples = 100, features=num_features, classes=num_classes)

    model = FFNsSparse3(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes,
                        num_hidden_layers=num_hidden_layers, sparsity=0.9)
    #model = DenseFFN(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes, num_hidden_layers=num_hidden_layers)

    t = model.trainable_variables
    train(model, X_train, y_train,epochs=500, batch_size=128,lr= 0.01,
          prune_and_regrow_step=155)


if __name__ == '__main__':
    main()

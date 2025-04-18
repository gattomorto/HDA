import os
import sys
from guppy import hpy
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_utils import *
from memory_profiler import profile
from memory_profiler import memory_usage
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
    # ascending
    idx = tf.argsort(tf.math.abs(momenta[i]))
    momentum_sorted = tf.gather(momenta[i], idx)
    indices_sorted = tf.gather(model.W_indices[i], idx)
    values_sorted = tf.gather(model.W_values[i], idx)

    split_idx = tf.shape(momentum_sorted)[0] // 2
    #split_idx = 3

    indices_new = indices_sorted[split_idx:]
    values_new = values_sorted[split_idx:]
    momentum_new = momentum_sorted[split_idx:]

    model.W_indices[i]=indices_new
    #model.W_values[i]=values_new
    #model.W_values[i].assign(values_new)
    model.W_values[i] = tf.Variable(values_new, name=f"W{i + 1}_values", trainable=True)

    return split_idx.numpy()


#TODO: fanno schifo i nomi e get_contributions ha degli zeri per layers che non sono in non_saturated_layer -- fa schifo
#TODO: layers in teeoria non serve pechè li ottieni da m
#TODO: passare direttamente il numero di elementi da ricrescere a regrow_layer piuttosto che la proporzione & to_regrow?!
#TODO: cambiare il tipo di eccezzione
def regrow(model, momenta, to_regrow, layers):
    to_regrow_iniziale = to_regrow
    nnz_before = get_total_nonzero_weights(model)

    while to_regrow != 0:
        prop_contributions = get_contributions(layers, momenta)
        tot_missing = 0
        # il nome non è proprio giusto dato che potrebbe capitare che il numero da allocare = allo spazio disponibile -- in questo caso l_missing = 0 ed entrerebbe in non_saturated_layers
        non_saturated_layers = []
        for l in layers:
            l_missing = regrow_layer(l, model, prop_contributions[l], to_regrow)
            tot_missing = tot_missing + l_missing
            if l_missing == 0: non_saturated_layers.append(l)
        layers = non_saturated_layers
        to_regrow = tot_missing
    nnz_after = get_total_nonzero_weights(model)

    if nnz_before + to_regrow_iniziale != nnz_after:
        raise Exception("regrown diverso da pruned")


#TODO: expected invece di theoretic?
#TODO: riordinare gli indici?!
def regrow_layer(i, model, contribution, missing):
    theoretic_num_regrow = round(missing * contribution)

    '''if theoretic_num_regrow == 0:
        raise ValueError(f"No regrowth")'''

    shape = model.W_shapes[i]
    all_coords = np.array([(r, c) for r in range(shape[0]) for c in range(shape[1])], dtype=np.int64)

    existing = set(map(tuple, model.W_indices[i].numpy()))
    mask = np.array([tuple(coord) not in existing for coord in all_coords])
    available_coords = all_coords[mask]
    num_available_coords = len(available_coords)

    if theoretic_num_regrow > num_available_coords:
        actual_num_regrow = num_available_coords
    else:
        actual_num_regrow = theoretic_num_regrow

    chosen = available_coords[np.random.choice(len(available_coords), size=actual_num_regrow, replace=False)]

    new_indices = tf.constant(chosen, dtype=tf.int64)
    new_values = tf.random.normal([actual_num_regrow])

    model.W_indices[i] = tf.concat([model.W_indices[i], new_indices], axis=0)
    model.W_values[i] = tf.Variable(tf.concat([model.W_values[i], new_values], axis=0), name=f"W{i + 1}_values",
                                    trainable=True)

    return theoretic_num_regrow-actual_num_regrow

def get_total_nonzero_weights(model):
    tot = 0
    for i in range(model.num_layers):
        tot = tot + get_num_nonzero_weights(i,model)
    return tot

def get_num_nonzero_weights(i,model):
    return tf.shape(model.W_indices[i])[0].numpy()


#TODO: cambia il tipo di eccezione
#TODO: riscrivi in forma vettoriale?
def get_contributions(layers, momenta):
    if layers==[]:
        raise Exception("layers empty, probably to_grow > available space")

    mean_momentas = [0 for _ in range(len(momenta))] #forse è meglio sostituire con model.num_layers
    for l in layers:
        mean_momentum = np.mean(np.abs(momenta[l]))
        mean_momentas[l] = mean_momentum

    total_momentum = sum(mean_momentas)

    momentum_contribution = []
    for mean_momentum in mean_momentas:
        proportion = mean_momentum / total_momentum
        momentum_contribution.append(proportion)

    return momentum_contribution


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
                pass
                #print(f"Step {step:03d} | Loss: {loss:.4f}")
                #print(mem_usage())
                #print("--- %s seconds ---" % (time.time() - start_time))

                #exit(-1)
                #print("W1_values:", model.W1_values.numpy())


            for var in optimizer.variables:
                if 'W' in var.path and 'momentum' in var.path:
                    momenta.append(var.value)

            tot_pruned = 0
            print("tot nonzero weights:",get_total_nonzero_weights(model))
            for i in range(model.num_layers):
                nnz = get_num_nonzero_weights(i, model)
                pruned = prune_layer(i,momenta,model)
                tot_pruned = tot_pruned + pruned
                #print(get_num_nonzero_weights(i, model))
                #print("layer:",i,",non zero weights:",nnz ,",pruned:",pruned)
            #print("tot pruned: ",tot_pruned)

            #print("tot nonzero weights: ",get_total_nonzero_weights(model))
            tot_regrown = 0

            regrow(model,momenta,tot_pruned, [l for l in range(model.num_layers)] )
            print("tot nonzero weights: ",get_total_nonzero_weights(model))








            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            momenta=[]


def main():
    num_features = 50
    num_classes = 2
    hidden_dim =  50
    num_hidden_layers = 1


    X, Y = generate_data(samples = 100, features=num_features, classes=num_classes)

    model = FFNsSparse3(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes,
                        num_hidden_layers=num_hidden_layers, sparsity=0.511)
    #model = DenseFFN(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes, num_hidden_layers=num_hidden_layers)

    t = model.trainable_variables
    train(model, X, Y,epochs=5, batch_size=10,lr= 0.001)


if __name__ == '__main__':
    main()

import os

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Must be set before importing TensorFlow!

import random
import sys
#from guppy import hpy
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from keras.src.backend.tensorflow.sparse import sparse_to_dense
from tensorflow.python.ops.gen_sparse_ops import sparse_reorder, sparse_tensor_dense_mat_mul, sparse_to_dense
import conv
import funzioni
#from memory_profiler import profile
#from memory_profiler import memory_usage

from utils import *
np.set_printoptions(threshold=np.inf)
SEED = 2
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#shape (m,n)
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

#shape (m,n) row major
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




def create_tensor_row_major_old(K, Cin, Cout, sparsity=0):
    # Total number of elements
    total_elements = K * K * Cin * Cout

    # Number of non-zero elements
    nnz = int(total_elements * (1 - sparsity))

    # Create a flat tensor with sequential values followed by zeros
    flat_values = tf.concat([
        tf.range(nnz, dtype=tf.float32),  # Non-zero values
        tf.zeros(total_elements - nnz, dtype=tf.float32)  # Zeros
    ], axis=0)

    # Shuffle the flat tensor to randomize positions
    shuffled_indices = tf.random.shuffle(tf.range(total_elements))
    flat_values = tf.gather(flat_values, shuffled_indices)

    # Reshape into [K, K, Cin, Cout] (row-major order)
    tensor = tf.reshape(flat_values, [K, K, Cin, Cout])

    #zeros = tf.equal(tensor, 0)
    #print( tf.reduce_sum(tf.cast(zeros, tf.int64)))

    return tensor

def create_tensor_row_major(K, Cin, Cout, sparsity=0):
    # Total number of elements
    total_elements = K * K * Cin * Cout

    # Number of non-zero elements
    nnz = int(total_elements * (1 - sparsity))

    # Create a flat tensor with sequential values (starting from 1) followed by zeros
    flat_values = tf.concat([
        tf.range(1, nnz + 1, dtype=tf.float32),  # Non-zero values (1, 2, ..., nnz)
        tf.zeros(total_elements - nnz, dtype=tf.float32)  # Zeros
    ], axis=0)

    SEED = 2
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    shuffled_indices = tf.random.shuffle(tf.range(total_elements))
    flat_values = tf.gather(flat_values, shuffled_indices)

    # Reshape into [K, K, Cin, Cout] (row-major order)
    tensor = tf.reshape(flat_values, [K, K, Cin, Cout])

    '''total_elements = tf.size(tensor, out_type=tf.int64)
    non_zeros = tf.math.count_nonzero(tensor)
    xxx =  total_elements - non_zeros'''

    return tensor


# è come FFNsSparse, tranne che si aspetta X (batch_size, input_dim)
#TODO: set indices, set values
#TODO: sono sicuro che indices deve essere tf.Constant anche se viene modificato in prune & regrow?
class FFNsSparse(tf.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, sparsity=0.0,option ='B'):
        super().__init__(name=None)
        self.option = option

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
            #13s 13 14 14 27 21 22
            if self.option == 'A1':
                W = tf.sparse.SparseTensor(indices=self.W_indices[i], values=self.W_values[i], dense_shape=self.W_shapes[i])
                out = tf.sparse.sparse_dense_matmul(out,W)
            #22 14 9 12 12 12 14 14 8 8 8 8 13 13 10 10 18 13 13 11
            elif self.option == 'B1':
                W = tf.sparse.SparseTensor(indices=self.W_indices[i], values=self.W_values[i], dense_shape=self.W_shapes[i])
                W = tf.sparse.to_dense(W,validate_indices=False)
                # b_is_sparse dà warning
                out = tf.matmul(out,W,b_is_sparse=True)
            #9s 11s 18s 12s 15s 14s 8 8 8 10 10 11 11 10 13
            elif self.option == 'B2':
                W = sparse_to_dense(self.W_indices[i], self.W_shapes[i], self.W_values[i],
                                     default_value=0,
                                     validate_indices=False)
                out = tf.matmul(out,W,b_is_sparse=True)

            out = out + self.b[i]
            if i < self.num_layers - 1:
                out = tf.nn.relu(out)

        return out
class FFNsSparse_REVERSED(tf.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, sparsity=0.0,option = 'B'):

        super().__init__(name=None)
        self.option = option
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

            indices, nnz = random_sparse_indices2(shape, sparsity=sparsity, seed=i)
            values = tf.Variable(tf.random.normal([nnz]), name=f"W{i}_values")
            bias = tf.Variable(tf.zeros([out_dim, 1]), name=f"b{i}")

            self.W_indices.append(indices)
            self.W_values.append(values)
            self.W_shapes.append(shape)
            self.b.append(bias)


    def __call__(self, X):
        out = X
        for i in range(self.num_layers):
            #13 17 13 14 14
            if self.option == 'A1':
                W = tf.sparse.SparseTensor(indices=self.W_indices[i], values=self.W_values[i], dense_shape=self.W_shapes[i])
                out = tf.sparse.sparse_dense_matmul(W, out)
            #22 15 14 14 14 14
            elif self.option == 'A2':
                out = sparse_tensor_dense_mat_mul(self.W_indices[i], self.W_values[i], self.W_shapes[i],out)
            # 14 13 13 12 12 13
            elif self.option == 'B1':
                W = tf.sparse.SparseTensor(indices=self.W_indices[i], values=self.W_values[i], dense_shape=self.W_shapes[i])
                W = tf.sparse.to_dense(W,validate_indices=False)
                out = tf.matmul(W, out,a_is_sparse=True)
            # 15 13 13 13 14 16 14
            elif self.option == 'B2':
                W = sparse_to_dense(self.W_indices[i], self.W_shapes[i],self.W_values[i],default_value=0,validate_indices=False)
                out = tf.matmul(W, out,a_is_sparse=True)


            out = out + self.b[i]
            if i < self.num_layers - 1:
                out = tf.nn.relu(out)

        return out
class DenseFFN(tf.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):

        super().__init__(name=None)

        self.num_layers = num_hidden_layers + 1

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
'''
#inizializza con create_tensor_row_major -- ConvDense e ConvSparse2 devono dare lo stesso risultato

class ConvDense(tf.Module):
    def __init__(self, num_classes=10):
        super().__init__(name=None)


        # ---- Convolutional Layer 1 ----
        in_chan1 = 1
        out_chan1 = 5
        #self.conv1_w = tf.Variable(tf.random.normal([3, 3, in_chan1, out_chan1], stddev=0.1), name="conv1_w")
        self.conv1_w =  tf.Variable(create_tensor_row_major(3,in_chan1,out_chan1), name="conv1_w")
        self.conv1_b =  tf.Variable(tf.zeros([out_chan1]), name="conv1_b")



        # ---- Convolutional Layer 2 ----
        in_chan2 = out_chan1
        out_chan2 = 8
        #self.conv2_w = tf.Variable( tf.random.normal([3, 3, in_chan2, out_chan2], stddev=0.1), name="conv2_w")
        self.conv2_w = tf.Variable(create_tensor_row_major(3,in_chan2,out_chan2), name="conv2_w")
        self.conv2_b = tf.Variable(tf.zeros([out_chan2]), name="conv2_b")

        # ---- Fully Connected Layers ----
        #self.fc1_w = tf.Variable(tf.random.normal([7*7*out_chan2, 128], stddev=0.1), name="fc1_w")
        self.fc1_w = tf.Variable(funzioni.SparseTensor([7*7*out_chan2, 128],0,name="fc1_wmio").to_tf_dense(),name="fc1_wmio")
        #self.fc1_w = tf.Variable(tf.ones(shape=(7 * 7 * 8, 128)),name = "fc1_w")
        self.fc1_b = tf.Variable(tf.zeros([128]), name="fc1_b")

        #self.fc2_w = tf.Variable(tf.random.normal([128, num_classes], stddev=0.1), name="fc2_w")
        self.fc2_w = tf.Variable(funzioni.SparseTensor([128, num_classes],0,name="fc2_wmio").to_tf_dense(),name="fc2_wmio")
        #self.fc2_w = tf.Variable(tf.ones(shape=(128, num_classes)),name = "fc2_w")
        self.fc2_b = tf.Variable(tf.zeros([num_classes]), name="fc2_b")

    def __call__(self, x):
        # x: [batch_size, 28, 28, 1]

        # ---- Conv Layer 1 + ReLU + MaxPool ----
        x = tf.nn.conv2d(x, self.conv1_w, strides=1, padding='SAME')

        x = tf.nn.bias_add(x, self.conv1_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')  # [batch, 14, 14, 32]

        # ---- Conv Layer 2 + ReLU + MaxPool ----
        x = tf.nn.conv2d(x, self.conv2_w, strides=1, padding='SAME')
        x = tf.nn.bias_add(x, self.conv2_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')  # [batch, 7, 7, 64]

        # ---- Flatten ----
        x = tf.reshape(x, [x.shape[0], -1])  # [batch, 7*7*64]


        # ---- Dense Layer + ReLU ----
        x = tf.matmul(x, self.fc1_w) + self.fc1_b

        x = tf.nn.relu(x)


        # ---- Output Layer (logits) ----
        logits = tf.matmul(x, self.fc2_w) + self.fc2_b
        #tf.io.write_file('de.bytes', tf.io.serialize_tensor(logits))
        #exit()
        return logits
class ConvSparse2(tf.Module):
    def __init__(self, num_classes=10):
        super().__init__(name=None)
        #sparsity = 0.7
        # ---- Convolutional Layer 1 ----

        self.conv1_shape = [3, 3, 1, 5]
        #self.conv1_indices, conv1_nnz = funzioni.SparseTensor.random_sparse_indices3(self.conv1_shape,sparsity)
        #self.conv1_values = tf.Variable(tf.random.normal([conv1_nnz]), name="conv1_values")
        #self.conv1_dense = tf.sparse.to_dense(tf.sparse.SparseTensor(self.conv1_indices, self.conv1_values, self.conv1_shape))
        #tt = create_tensor_row_major(3,1,5)
        #tt_sp = tf.sparse.from_dense(tt)
        #self.conv1_values = tf.Variable(tt_sp.values, name="conv1_values")
        #self.conv1_indices = tt_sp.indices
        #tt = tf.reshape(tt,-1)
        #self.conv1_values = tf.Variable(tt, name="conv1_values")
        self.conv1_w =tf.recompute_grad( funzioni.SparseTensor(create_tensor_row_major(3,1,5)))
        #self.conv1_w = funzioni.SparseTensor(self.conv1_shape,sparsity,name = "conv1_w")
        #xxx = self.conv1_w.to_tf_dense()
        #print(tf.reduce_max(tf.abs(tt - xxx)))
        self.conv1_b = tf.Variable(tf.zeros([5]), name="conv1_b")

        # ---- Convolutional Layer 2 ----
        #self.conv2_w = tf.Variable(tf.random.normal([3, 3, 5, 8], stddev=0.1), name="conv2_w")
        self.conv2_shape = [3, 3, 5, 8]
        #self.conv2_indices, conv2_nnz = funzioni.SparseTensor.random_sparse_indices3(self.conv2_shape,sparsity)
        #self.conv2_values = tf.Variable(tf.random.normal([conv2_nnz]), name="conv2_values")
        #tt = create_tensor_row_major(3, 5, 8)
        #tt_sp = tf.sparse.from_dense(tt)
        #self.conv2_values = tf.Variable(tt_sp.values, name="conv2_values")
        #self.conv2_indices = tt_sp.indices
        self.conv2_w = funzioni.SparseTensor(create_tensor_row_major(3, 5, 8))
        #self.conv2_w = funzioni.SparseTensor(self.conv2_shape,sparsity, name = "conv2_w")
        #xxx = self.conv2_w.to_tf_dense()
        # print(tf.reduce_max(tf.abs(tt - xxx)))
        #tt = tf.reshape(tt, -1)
        #self.conv2_values = tf.Variable(tt, name="conv2_values")
        self.conv2_b = tf.Variable(tf.zeros([8]), name="conv2_b")


        # ---- Fully Connected Layers ----
        #self.fc1_w = tf.Variable(tf.random.normal([7 * 7 * 8, 128], stddev=0.1), name="fc1_w")
        self.fc1_w = funzioni.SparseTensor([7 * 7 * 8, 128],0 ,name="fc1_wmio")
        #self.fc1_w = funzioni.SparseTensor(tf.ones(shape=[7 * 7 * 8, 128]),name ="fc1_wmio")
        self.fc1_b = tf.Variable(tf.zeros([128]), name="fc1_b")


        #self.fc2_w = tf.Variable(tf.random.normal([128, num_classes], stddev=0.1), name="fc2_w")
        #self.fc2_w = funzioni.SparseTensor(tf.ones(shape=[128, num_classes]),name ="fc2_wmio")
        self.fc2_w = funzioni.SparseTensor([128, num_classes],0,name="fc2_wmio")
        self.fc2_b = tf.Variable(tf.zeros([num_classes]), name="fc2_b")

    def __call__(self, x):
        # x: [batch_size, 28, 28, 1]

        # ---- Conv Layer 1 + ReLU + MaxPool ----
        #conv1_sparse_filter = tf.SparseTensor(self.conv1_indices, self.conv1_values, self.conv1_shape)
        #conv1_sparse_dense_filter = tf.sparse.to_dense(conv1_sparse_filter)

        #print(tf.reduce_max(tf.abs(conv1_sparse_dense_filter - self.conv1_dense)))

        #x= tf.nn.conv2d(x,conv1_sparse_dense_filter, strides=1, padding='SAME')
        x = conv.sparse_to_dense_conv2d(x,self.conv1_w, stride=1, padding='SAME')



        x = tf.nn.bias_add(x, self.conv1_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')  # [batch, 14, 14, 32]


        # ---- Conv Layer 2 + ReLU + MaxPool ----
        # x = tf.nn.conv2d(x, self.conv2_w, strides=1, padding='SAME')
        #conv2_sparse_filter = tf.sparse.from_dense(self.conv2_w)
        #conv2_sparse_filter = tf.SparseTensor(self.conv2_indices, self.conv2_values, self.conv2_shape)
        #conv2_sparse_dense_filter = tf.sparse.to_dense(conv2_sparse_filter)
        #x = tf.nn.conv2d(x,conv2_sparse_dense_filter, strides=1, padding='SAME')
        x = conv.sparse_to_dense_conv2d(x,self.conv2_w, stride=1, padding='SAME')

        x = tf.nn.bias_add(x, self.conv2_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')  # [batch, 7, 7, 64]


        # ---- Flatten ----
        x = tf.reshape(x, [x.shape[0], -1])  # [batch, 7*7*64]



        # ---- Dense Layer + ReLU ----
        #x = tf.matmul(x, self.fc1_w) + self.fc1_b
        x = conv.matmul(x, self.fc1_w) + self.fc1_b
        x = tf.nn.relu(x)

        # ---- Output Layer (logits) ----
        logits = conv.matmul(x, self.fc2_w) + self.fc2_b
        #tf.io.write_file('sp.bytes', tf.io.serialize_tensor(logits))
        #exit()

        return logits
'''
class ConvDense(tf.Module):
    def __init__(self, num_classes=10):
        super().__init__(name=None)


        # ---- Convolutional Layer 1 ----
        in_chan1 = 1
        out_chan1 = 5
        self.conv1_w = tf.Variable(tf.random.normal([3, 3, in_chan1, out_chan1], stddev=0.1), name="conv1_w")
        #self.conv1_w =  tf.Variable(create_tensor_row_major(3,in_chan1,out_chan1), name="conv1_w")
        self.conv1_b =  tf.Variable(tf.zeros([out_chan1]), name="conv1_b")



        # ---- Convolutional Layer 2 ----
        in_chan2 = out_chan1
        out_chan2 = 8
        self.conv2_w = tf.Variable( tf.random.normal([3, 3, in_chan2, out_chan2], stddev=0.1), name="conv2_w")
        #self.conv2_w = tf.Variable(create_tensor_row_major(3,in_chan2,out_chan2), name="conv2_w")
        self.conv2_b = tf.Variable(tf.zeros([out_chan2]), name="conv2_b")

        # ---- Fully Connected Layers ----
        self.fc1_w = tf.Variable(tf.random.normal([7*7*out_chan2, 128], stddev=0.1), name="fc1_w")
        #self.fc1_w = tf.Variable(funzioni.SparseTensor([7*7*out_chan2, 128],0,name="fc1_wmio").to_tf_dense(),name="fc1_wmio")
        #self.fc1_w = tf.Variable(tf.ones(shape=(7 * 7 * 8, 128)),name = "fc1_w")
        self.fc1_b = tf.Variable(tf.zeros([128]), name="fc1_b")

        self.fc2_w = tf.Variable(tf.random.normal([128, num_classes], stddev=0.1), name="fc2_w")
        #self.fc2_w = tf.Variable(funzioni.SparseTensor([128, num_classes],0,name="fc2_wmio").to_tf_dense(),name="fc2_wmio")
        #self.fc2_w = tf.Variable(tf.ones(shape=(128, num_classes)),name = "fc2_w")
        self.fc2_b = tf.Variable(tf.zeros([num_classes]), name="fc2_b")

    def __call__(self, x):
        # x: [batch_size, 28, 28, 1]

        # ---- Conv Layer 1 + ReLU + MaxPool ----
        x = tf.nn.conv2d(x, self.conv1_w, strides=1, padding='SAME')
        x = tf.nn.bias_add(x, self.conv1_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')  # [batch, 14, 14, 32]

        # ---- Conv Layer 2 + ReLU + MaxPool ----
        x = tf.nn.conv2d(x, self.conv2_w, strides=1, padding='SAME')
        x = tf.nn.bias_add(x, self.conv2_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')  # [batch, 7, 7, 64]

        # ---- Flatten ----
        x = tf.reshape(x, [x.shape[0], -1])  # [batch, 7*7*64]


        # ---- Dense Layer + ReLU ----
        x =  tf.matmul(x, self.fc1_w) + self.fc1_b

        x = tf.nn.relu(x)


        # ---- Output Layer (logits) ----
        logits = tf.matmul(x, self.fc2_w) + self.fc2_b
        #tf.io.write_file('de.bytes', tf.io.serialize_tensor(logits))
        #exit()
        return logits

# usa funzioni.SparseTensor
class ConvSparse(tf.Module):
    def __init__(self, sparsity ,num_classes=10):
        super().__init__(name=None)
        # ---- Convolutional Layer 1 ----


        self.conv1_w = funzioni.SparseTensor([3, 3, 1, 5],sparsity)
        self.conv1_b = tf.Variable(tf.zeros([5]), name="conv1_b")


        self.conv2_w = funzioni.SparseTensor([3, 3, 5, 8], sparsity)
        self.conv2_b = tf.Variable(tf.zeros([8]), name="conv2_b")



        self.fc1_w = funzioni.SparseTensor([7 * 7 * 8, 128],sparsity,name="fc1_wmio")
        self.fc1_b = tf.Variable(tf.zeros([128]), name="fc1_b")

        self.fc2_w = funzioni.SparseTensor([128, num_classes],sparsity,name="fc2_wmio")
        self.fc2_b = tf.Variable(tf.zeros([num_classes]), name="fc2_b")

    def __call__(self, x):
        x = conv.sparse_to_dense_conv2d(x,self.conv1_w, stride=1, padding='SAME')

        x = tf.nn.bias_add(x, self.conv1_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        x = conv.sparse_to_dense_conv2d(x,self.conv2_w, stride=1, padding='SAME')
        x = tf.nn.bias_add(x, self.conv2_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')  # [batch, 7, 7, 64]


        x = tf.reshape(x, [x.shape[0], -1])  # [batch, 7*7*64]

        x = conv.matmul(x, self.fc1_w) + self.fc1_b
        x = tf.nn.relu(x)

        logits = conv.matmul(x, self.fc2_w) + self.fc2_b


        return logits

class ConvSparse_check(tf.Module):
    def __init__(self, sparsity, num_classes=10):
        super().__init__(name=None)

        self.conv1_w = funzioni.SparseTensor([3, 3, 1, 5],sparsity)
        self.conv1_b = tf.Variable(tf.zeros([5]), name="conv1_b")

        self.conv2_w = funzioni.SparseTensor([3, 3, 5, 8],sparsity)
        self.conv2_b = tf.Variable(tf.zeros([8]), name="conv2_b")

        self.fc1_w = funzioni.SparseTensor([7 * 7 * 8, 128], sparsity, name="fc1_wmio")
        self.fc1_b = tf.Variable(tf.zeros([128]), name="fc1_b")

        self.fc2_w = funzioni.SparseTensor([128, num_classes], sparsity, name="fc2_wmio")
        self.fc2_b = tf.Variable(tf.zeros([num_classes]), name="fc2_b")

    def conv_block_1(self, x):
        #@tf.recompute_grad
        def block(x):
            x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=1, padding='SAME')
            x = tf.nn.bias_add(x, self.conv1_b)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
            return x
        return block(x)

    def conv_block_2(self, x):
        #@tf.recompute_grad
        def block(x):
            x =  conv.sparse_to_dense_conv2d(x, self.conv2_w, stride=1, padding='SAME')
            x = tf.nn.bias_add(x, self.conv2_b)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
            return x
        return block(x)

    def __call__(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        x = tf.reshape(x, [tf.shape(x)[0], -1])
        x = conv.matmul(x, self.fc1_w) + self.fc1_b
        x = tf.nn.relu(x)

        logits = conv.matmul(x, self.fc2_w) + self.fc2_b
        return logits

# usa indices , values, e shape senza dichiarare funzioni.SparseTensor
class ConvSparse_explicit(tf.Module):
    def __init__(self, sparsity ,num_classes=10):
        super().__init__(name=None)
        # ---- Convolutional Layer 1 ----


        #self.conv1_w = funzioni.SparseTensor([3, 3, 1, 5],sparsity)
        self.conv1_shape = [3, 3, 1, 5]
        self.conv1_indices,nnz = funzioni.SparseTensor.random_sparse_indices3(self.conv1_shape,sparsity)
        self.conv1_values = tf.Variable(tf.random.normal([nnz]), name="conv1_values")
        self.conv1_b = tf.Variable(tf.zeros([5]), name="conv1_b")


        #self.conv2_w = funzioni.SparseTensor([3, 3, 5, 8], sparsity)
        self.conv2_shape = [3, 3, 5, 8]
        self.conv2_indices,nnz = funzioni.SparseTensor.random_sparse_indices3(self.conv2_shape,sparsity)
        self.conv2_values = tf.Variable(tf.random.normal([nnz]), name="conv2_values")
        self.conv2_b = tf.Variable(tf.zeros([8]), name="conv2_b")


        #self.fc1_w = funzioni.SparseTensor([7 * 7 * 8, 128],sparsity,name="fc1_wmio")
        self.fc1_shape = [7 * 7 * 8, 128]
        self.fc1_indices,nnz = funzioni.SparseTensor.random_sparse_indices3(self.fc1_shape,sparsity)
        self.fc1_values = tf.Variable(tf.random.normal([nnz]), name="fc1_values")
        self.fc1_b = tf.Variable(tf.zeros([128]), name="fc1_b")

        #self.fc2_w = funzioni.SparseTensor([128, num_classes],sparsity,name="fc2_wmio")
        self.fc2_shape = [ 128, num_classes]
        self.fc2_indices,nnz = funzioni.SparseTensor.random_sparse_indices3(self.fc2_shape,sparsity)
        self.fc2_values = tf.Variable(tf.random.normal([nnz]), name="fc2_values")
        self.fc2_b = tf.Variable(tf.zeros([num_classes]), name="fc2_b")

    def __call__(self, x):
        #x = conv.sparse_to_dense_conv2d(x,self.conv1_w, stride=1, padding='SAME')
        conv1_sparse = tf.sparse.SparseTensor(self.conv1_indices, self.conv1_values, self.conv1_shape)
        conv1_dense = tf.sparse.to_dense(conv1_sparse)
        x = tf.nn.conv2d(x,conv1_dense,strides=1,padding='SAME')

        x = tf.nn.bias_add(x, self.conv1_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
#----------------------------------------------------------------------------------------------
        #x = conv.sparse_to_dense_conv2d(x,self.conv2_w, stride=1, padding='SAME')

        conv2_sparse = tf.sparse.SparseTensor(self.conv2_indices, self.conv2_values, self.conv2_shape)
        conv2_dense = tf.sparse.to_dense(conv2_sparse)
        x = tf.nn.conv2d(x,conv2_dense,strides=1,padding='SAME')


        x = tf.nn.bias_add(x, self.conv2_b)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')  # [batch, 7, 7, 64]


        x = tf.reshape(x, [x.shape[0], -1])  # [batch, 7*7*64]

        #x = conv.matmul(x, self.fc1_w) + self.fc1_b

        fc1_sparse = tf.sparse.SparseTensor(self.fc1_indices, self.fc1_values, self.fc1_shape)
        fc1_dense = tf.sparse.to_dense(fc1_sparse)
        x = tf.matmul(x,fc1_dense) + self.fc1_b

        x = tf.nn.relu(x)

        #logits = conv.matmul(x, self.fc2_w) + self.fc2_b

        fc2_sparse = tf.sparse.SparseTensor(self.fc2_indices, self.fc2_values, self.fc2_shape)
        fc2_dense = tf.sparse.to_dense(fc2_sparse)
        logits = tf.matmul(x,fc2_dense) + self.fc2_b



        return logits



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

#TODO: cosa fare con import? perchè non ce tf. davanti a sparse_reorder
#TODO: reorder_weights
def reorder_indices_and_values(model,i):
    indices = model.W_indices[i]
    values = model.W_values[i]
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
    start_time = time.perf_counter()

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
            #print(step)
            if it % 1 == 0:##
                pass

                acc = test(model,X,y)
                #acc = test_transpose(model,X,y)
                #print(f"it: {it}, acc: {acc:.3f}")
                print(f"Step {step:03d} | Loss: {loss}")
                print(mem_usage())
                #print(time.perf_counter() - start_time, "seconds")

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

def test_transpose(model, X, y):
    X_t = tf.transpose(X, perm=[1, 0])
    logits = model(X_t)
    logits = tf.transpose(logits, perm=[1, 0])  # per confronto con y
    correct_preds = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    return acc.numpy()

#TODO: capire. perchè hidden_dim = 1 dà problemi
def main():
    num_features = 784
    num_classes = 10
    hidden_dim = 1000
    num_hidden_layers = 50
    (X_train,y_train),(X_test,y_test)= load_mnist_data(flatten=True)

    #X_train, y_train = generate_data(samples = 100, features=num_features, classes=num_classes)
    # (1037.427734375, 3289.36328125, 2622.01171875, 3471.2890625) sparse ffn B1/B2
    # (2401.93798828125, 5394.37109375, 3722.74609375, 5752.55859375) sparse ffn A1
    # (1481.763671875, 3174.77734375, 2290.27734375, 3187.3359375) dense ffn
    model = FFNsSparse(input_dim=num_features,hidden_dim=hidden_dim,output_dim=num_classes,num_hidden_layers=num_hidden_layers,sparsity=0.9,option='A1')
    #model = DenseFFN(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes, num_hidden_layers=num_hidden_layers)
    #model= ConvSparse_check(num_classes= 10)
    #model= ConvDense_original(num_classes= 10)
    #dmodel= ConvSparse_explicit(sparsity=0,num_classes= 10)

    '''sp = tf.io.parse_tensor(tf.io.read_file('sp.bytes'), out_type=tf.float32)
    de = tf.io.parse_tensor(tf.io.read_file('de.bytes'), out_type=tf.float32)
    print(tf.reduce_max(tf.abs(sp - de)))
    exit()'''

    t = model.trainable_variables
    trainable_count = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print("Total number of trainable scalars:", trainable_count)
    train(model, X_train, y_train,epochs=500, batch_size=2048,lr= 0.01,
          prune_and_regrow_step=3000)


if __name__ == '__main__':
    main()

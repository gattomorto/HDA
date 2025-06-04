folder_path = '/content/drive/MyDrive/hda'
import sys
sys.path.append(folder_path)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from tensorflow import recompute_grad
import funzioni
import tensorflow as tf
import random
import numpy as np
import v4
import conv
import utils
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D

import gc
#tf.keras.backend.set_floatx('float16')



SEED = 0
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#TODO: torch.no_grad()
#TODO: tf.config.experimental.reset_memory_stats(gpus[0])


class SparseTensor(tf.Module):
    # shape generico row major
    #TODO: cambia in sparse_indices_init & metti in init & rng & usare solo operazioni tf
    #TODO: capire se devono essere ordinati
    @staticmethod
    def random_sparse_indices3(shape, sparsity):
        SEED = 2
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        total = np.prod(shape)
        nnz = int((1.0 - sparsity) * total)  # Number of non-zero elements

        rng = np.random.default_rng(0)
        chosen = rng.choice(total, size=nnz, replace=False)

        # Convert flat indices to multi-dimensional indices
        unraveled = np.stack(np.unravel_index(chosen, shape), axis=-1)  # shape: (nnz, len(shape))

        # Convert multi-dimensional indices back to flat indices in row-major order for sorting
        flat_sorted_order = np.ravel_multi_index(unraveled.T, shape)
        sorted_indices = unraveled[np.argsort(flat_sorted_order)]

        return tf.constant(sorted_indices, dtype=tf.int64), nnz

    def __init__(self, *args,  name=None):
        SEED = 2
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        super().__init__(name=name)
        self.relative_momentum_contribution = 0
        self.mean_momentum = 0
        self.num_pruned = 0
        self.num_regrown = 0
        self.num_active_weights_before_pruning=None

        if len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], (float,int)):
            shape, sparsity = args
            self.sparsity = sparsity
            self.shape = tf.convert_to_tensor(shape,dtype=tf.int64)
            self.indices, nnz = self.random_sparse_indices3(shape, sparsity)
            initializer = tf.keras.initializers.HeNormal()
            self.values = tf.Variable(initializer(shape=[nnz]), name=name)

        elif len(args) == 1 and isinstance(args[0], tf.Tensor):
            dense_tensor = args[0]
            tf_sparse = tf.sparse.from_dense(dense_tensor)
            self.shape = tf_sparse.dense_shape
            self.indices = tf_sparse.indices
            self.values = tf.Variable(tf_sparse.values, name=name)

        elif len(args) == 1 and isinstance(args[0], tf.SparseTensor):
            tf_sparse = args[0]
            self.shape = tf_sparse.dense_shape
            self.indices = tf_sparse.indices
            self.values = tf.Variable(tf_sparse.values, name=name)

        else:
            raise TypeError("Invalid arguments for SparseTensor initialization. "
                            "Expected either (shape: tuple, sparsity: float) or (dense_tensor: tf.Tensor).")

    def __eq__(self, other):
        return tf.reduce_all(tf.equal(self.to_tf_dense(), other.to_tf_dense()))

    def __str__(self):
        dense = self.to_tf_dense()
        return str(dense.numpy())

    def __repr__(self):
        return str(self)

    def rowmajor_reorder(self):
        from tensorflow.python.ops.gen_sparse_ops import sparse_reorder, sparse_tensor_dense_mat_mul, sparse_to_dense
        '''tf_sparse = self.to_tf_sparse()
        tf_sparse_reordered =  tf.sparse.reorder(tf_sparse)
        self.indices = tf_sparse_reordered.indices
        self.values.assign(tf_sparse_reordered.values)'''
        xx = sparse_reorder(self.indices, self.values, self.shape)
        self.indices = xx.output_indices
        self.values.assign(xx.output_values)

    def to_tf_sparse(self):
        return tf.sparse.SparseTensor(self.indices, self.values, self.shape)

    def to_tf_dense(self):
        from tensorflow.python.ops.gen_sparse_ops import sparse_reorder, sparse_tensor_dense_mat_mul, sparse_to_dense
        #return tf.sparse.to_dense(self.to_tf_sparse())
        return sparse_to_dense(self.indices,self.shape,self.values,default_value=tf.constant(0.0),validate_indices=True)

    #TODO: bisogna togliere i vecchi values da oggetti tracciati?
    def prune(self, rho):
        '''
        :param rho: fraction of nonzero weights to prune
        :return: number of pruned weights
        '''

        self.num_active_weights_before_pruning = self.num_active_weights()

        # Sort by absolute value
        idx_sorted = tf.argsort(tf.math.abs(self.values))
        values_sorted = tf.gather(self.values, idx_sorted)
        indices_sorted = tf.gather(self.indices, idx_sorted)

        split_idx = tf.cast(tf.math.ceil(tf.cast(tf.shape(values_sorted)[0], tf.float32) * rho), tf.int32)

        if split_idx==0:
            return 0

        #ATTENZIONE: gli indici potrebbero essere disordinati
        new_indices = indices_sorted[split_idx:]
        new_values = values_sorted[split_idx:]

        self.indices = new_indices
        #TODO: name
        self.values = tf.Variable(new_values, name="pruned_values", trainable=True)

        num_pruned = split_idx.numpy()
        self.num_pruned = self.num_pruned + num_pruned
        return num_pruned

    def regrow_original(self, requested_growth):
        # Generate all possible indices
        shape = self.shape
        all_coords = tf.stack(tf.meshgrid(*[tf.range(s) for s in shape], indexing='ij'), axis=-1)
        all_coords = tf.reshape(all_coords, [-1, len(shape)])

        # Convert existing indices to set of tuples
        existing = set(map(tuple, self.indices.numpy()))  # NOTE: still uses .numpy() here

        # Filter out existing indices
        mask = tf.constant([tuple(coord.numpy()) not in existing for coord in tf.unstack(all_coords, axis=0)])
        available_coords = tf.boolean_mask(all_coords, mask)

        num_available_coords = tf.shape(available_coords)[0]
        actual_regrow = tf.minimum(requested_growth, num_available_coords)

        # Select new indices randomly
        shuffled = tf.random.shuffle(available_coords)
        new_indices = shuffled[:actual_regrow]

        # Initialize new values (zeros or another init)
        new_values = tf.zeros([actual_regrow], dtype=self.values.dtype)

        # Update indices and values
        self.indices = tf.concat([self.indices, new_indices], axis=0)
        self.values = tf.Variable(tf.concat([self.values, new_values], axis=0), name="regrown_values", trainable=True)

        self.rowmajor_reorder()

        self.num_regrown = self.num_regrown + actual_regrow.numpy()

        return actual_regrow.numpy()

    def regrow(self, requested_growth):
        # Convert existing indices to a dense boolean mask
        existing_mask = tf.scatter_nd(
            self.indices,
            tf.ones(tf.shape(self.indices)[0], dtype=tf.bool),
            self.shape
        )

        # Find available positions by inverting the mask
        available_mask = tf.logical_not(existing_mask)
        available_coords = tf.where(available_mask)

        num_available = tf.shape(available_coords)[0]
        actual_regrow = tf.minimum(requested_growth, num_available)

        # Randomly sample from available coordinates
        if actual_regrow > 0:
            # Random sampling without replacement
            shuffled_indices = tf.random.shuffle(tf.range(num_available))
            selected_indices = shuffled_indices[:actual_regrow]
            new_indices = tf.gather(available_coords, selected_indices)

            # Initialize new values
            new_values = tf.zeros([actual_regrow], dtype=self.values.dtype)

            # Update indices and values
            self.indices = tf.concat([self.indices, new_indices], axis=0)

            #TODO: cambia name
            self.values = tf.Variable(
                tf.concat([self.values, new_values], axis=0),
                name="regrown_values",
                trainable=True
            )

            # Convert to numpy only once at the end
            actual_regrow_np = actual_regrow.numpy()
            self.num_regrown = self.num_regrown + actual_regrow_np
        else:
            actual_regrow_np = 0

        self.rowmajor_reorder()
        return actual_regrow_np

    def regrow_cl(self, requested_growth):
        if requested_growth == 0:
            return 0

        existing_set = set(map(tuple, self.indices.numpy()))
        shape_np = self.shape.numpy()
        total_elements = np.prod(shape_np)

        # Use numpy's random choice without replacement on a large range
        max_samples = min(total_elements, requested_growth * 5)  # Sample more than needed
        flat_indices = np.random.choice(total_elements, size=max_samples, replace=False)

        new_coords = []
        for flat_idx in flat_indices:
            coord = np.unravel_index(flat_idx, shape_np)
            if coord not in existing_set:
                new_coords.append(coord)
                if len(new_coords) >= requested_growth:
                    break

        if new_coords:
            new_indices_tf = tf.constant(new_coords, dtype=tf.int64)
            new_values = tf.zeros([len(new_coords)], dtype=self.values.dtype)

            self.indices = tf.concat([self.indices, new_indices_tf], axis=0)
            self.values = tf.Variable(
                tf.concat([self.values, new_values], axis=0),
                name="regrown_values", trainable=True
            )

            self.rowmajor_reorder()
            self.num_regrown += len(new_coords)
            return len(new_coords)

        return 0

    def reset_prune_and_regrow_stats(self):
        self.num_regrown = 0
        self.num_pruned = 0
        #self.num_active_weights_before_pruning

    def num_inactive_weights(self):
        return self.num_weights() - self.num_active_weights()

    def num_weights(self):
        return tf.reduce_prod(self.shape).numpy()

    def num_active_weights(self):
        return self.indices.shape[0]

    def is_saturated(self):
        return self.num_inactive_weights() == 0

    def tensor_sparsity(self):
        return self.num_inactive_weights()/self.num_weights()

# checkpointing parziale senza flag
'''
class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut

            self.w1 = SparseTensor([1, 1, in_channels, filters], sparsity, name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")

            self.w2 = SparseTensor([3, 3, filters, filters], sparsity, name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")

            self.w3 = SparseTensor([1, 1, filters, 4 * filters], sparsity, name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = SparseTensor([1, 1, in_channels, 4 * filters], sparsity, name="w_sc_M")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):

            @tf.recompute_grad
            def forward(x):
                shortcut = x
                if self.conv_shortcut:
                    shortcut = conv.sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                    shortcut = self.bn_sc(shortcut, training=training)

                x = conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1
                x = self.bn1(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2
                x = self.bn2(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3
                x = self.bn3(x, training=training)

                x = tf.nn.relu(x + shortcut)
                return x

            return forward(x)

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, sparsity, name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(
                ResNet50_sparse2.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, name="block1")
            )
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_sparse2.ConvBlock(sparsity, 4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:
                x = block(x, training=training)
            return x

    def __init__(self, sparsity, num_classes=8, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes

        self.conv1_w = SparseTensor([7, 7, 3, 64], sparsity, name="conv1_w_M")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, name="conv5")

        self.fc_w = SparseTensor([2048, num_classes], sparsity, name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

    def __call__(self, x, training=False):
        x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)

        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling

        logits = conv.matmul(x, self.fc_w) + self.fc_b
        return tf.nn.softmax(logits)
'''

#resnet sparso base
'''
class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            #initializer = tf.keras.initializers.HeNormal()

            # Conv layers
            #self.w1 = tf.Variable(initializer([1, 1, in_channels, filters]), name="w1")
            #self.w1 = tf.recompute_grad(funzioni.SparseTensor([1, 1, in_channels, filters],sparsity,name="w1_M"))
            self.w1 = SparseTensor([1, 1, in_channels, filters],sparsity,name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
            #-----------------------------------------------------------------
            #self.w2 = tf.Variable(initializer([3, 3, filters, filters]), name="w2")
            self.w2 = SparseTensor([3, 3, filters, filters],sparsity,name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
            #--------------------------------------------------------------------
            #self.w3 = tf.Variable(initializer([1, 1, filters, 4 * filters]), name="w3")
            self.w3 = SparseTensor([1, 1, filters, 4 * filters],sparsity,name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                #self.w_sc = tf.Variable(initializer([1, 1, in_channels, 4 * filters]), name="w_sc")
                self.w_sc = SparseTensor([1, 1, in_channels, 4 * filters],sparsity, name="w_sc_M")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        #@tf.recompute_grad
        def __call__(self, x, training=False):

            shortcut = x
            if self.conv_shortcut:
                #shortcut = tf.nn.conv2d(x, self.w_sc, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.b_sc
                shortcut = conv.sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                shortcut = self.bn_sc(shortcut, training=training)

            #x = tf.nn.conv2d(x, self.w1, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.b1
            #x =  tf.recompute_grad(conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME")) + self.b1
            x = conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1

            x = self.bn1(x, training=training)

            x = tf.nn.relu(x)

            #---------------------------------------------------
            #x = tf.nn.conv2d(x, self.w2, strides=[1, 1, 1, 1], padding="SAME") + self.b2
            x = conv.sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2

            x = self.bn2(x, training=training)

            x = tf.nn.relu(x)

            #---------------------------------------------------

            #x = tf.nn.conv2d(x, self.w3, strides=[1, 1, 1, 1], padding="SAME") + self.b3
            x = conv.sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3

            x = self.bn3(x, training=training)

            x = tf.nn.relu(x + shortcut)

            return x

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, sparsity ,name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(ResNet50_sparse2.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, name="block1"))
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_sparse2.ConvBlock(sparsity,4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:

                x = block(x, training=training)
            return x

    def __init__(self, sparsity, num_classes=8, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes

        #initializer = tf.keras.initializers.HeNormal()

        # Initial conv
        #self.conv1_w = tf.Variable(initializer([7, 7, 3, 64]), name="conv1_w")
        self.conv1_w = SparseTensor([7, 7, 3, 64],sparsity,name = "conv1_w_M")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        # Residual stages
        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity,name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2,  sparsity=sparsity,name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2,  sparsity=sparsity,name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, name="conv5")

        # Final FC layer
        #self.fc_w = tf.Variable(initializer([2048, num_classes]), name="fc_w")
        self.fc_w = SparseTensor([2048, num_classes],sparsity ,name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

    def __call__(self, x, training=False):
        #tf.io.write_file('sp.bytes', tf.io.serialize_tensor(x))
        #exit()
        #x = tf.nn.conv2d(x, self.conv1_w, strides=[1, 2, 2, 1], padding="SAME") + self.conv1_b
        x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b

        x = self.bn1(x, training=training)

        x = tf.nn.relu(x)

        x = self.pool(x)


        x = self.stage2(x, training=training)

        x = self.stage3(x, training=training)

        x = self.stage4(x, training=training)

        x = self.stage5(x, training=training)


        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling


        #logits = tf.matmul(x, self.fc_w) + self.fc_b
        logits = conv.matmul(x, self.fc_w) + self.fc_b


        return tf.nn.softmax(logits)
'''

# prune and regrow
'''
class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            #initializer = tf.keras.initializers.HeNormal()

            # Conv layers
            #self.w1 = tf.Variable(initializer([1, 1, in_channels, filters]), name="w1")
            #self.w1 = tf.recompute_grad(SparseTensor([1, 1, in_channels, filters],sparsity,name="w1_M"))
            self.w1 = SparseTensor([1, 1, in_channels, filters],sparsity,name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
            #-----------------------------------------------------------------
            #self.w2 = tf.Variable(initializer([3, 3, filters, filters]), name="w2")
            self.w2 = SparseTensor([3, 3, filters, filters],sparsity,name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
            #--------------------------------------------------------------------
            #self.w3 = tf.Variable(initializer([1, 1, filters, 4 * filters]), name="w3")
            self.w3 = SparseTensor([1, 1, filters, 4 * filters],sparsity,name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                #self.w_sc = tf.Variable(initializer([1, 1, in_channels, 4 * filters]), name="w_sc")
                self.w_sc = SparseTensor([1, 1, in_channels, 4 * filters],sparsity, name="w_sc_M")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")
        def __call__(self, x, training=False):

            shortcut = x
            if self.conv_shortcut:
                #shortcut = tf.nn.conv2d(x, self.w_sc, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.b_sc
                shortcut = conv.sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                shortcut = self.bn_sc(shortcut, training=training)

            #x = tf.nn.conv2d(x, self.w1, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.b1
            #x =  tf.recompute_grad(conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME")) + self.b1
            x = conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1

            x = self.bn1(x, training=training)

            x = tf.nn.relu(x)

            #---------------------------------------------------
            #x = tf.nn.conv2d(x, self.w2, strides=[1, 1, 1, 1], padding="SAME") + self.b2
            x = conv.sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2

            x = self.bn2(x, training=training)

            x = tf.nn.relu(x)

            #---------------------------------------------------

            #x = tf.nn.conv2d(x, self.w3, strides=[1, 1, 1, 1], padding="SAME") + self.b3
            x = conv.sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3

            x = self.bn3(x, training=training)

            x = tf.nn.relu(x + shortcut)

            return x
    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, sparsity ,name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(ResNet50_sparse2.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, name="block1"))
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_sparse2.ConvBlock(sparsity,4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:

                x = block(x, training=training)
            return x
    def __init__(self, sparsity, num_classes=8, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes

        #initializer = tf.keras.initializers.HeNormal()

        # Initial conv
        #self.conv1_w = tf.Variable(initializer([7, 7, 3, 64]), name="conv1_w")
        self.conv1_w = SparseTensor([7, 7, 3, 64],sparsity,name = "conv1_w_M")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        # Residual stages
        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity,name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2,  sparsity=sparsity,name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2,  sparsity=sparsity,name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, name="conv5")

        # Final FC layer
        #self.fc_w = tf.Variable(initializer([2048, num_classes]), name="fc_w")
        self.fc_w = SparseTensor([2048, num_classes],sparsity ,name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

        self.sparse_tensors = self.get_sparse_tensors()
    def __call__(self, x, training=False):
        #tf.io.write_file('sp.bytes', tf.io.serialize_tensor(x))
        #exit()
        #x = tf.nn.conv2d(x, self.conv1_w, strides=[1, 2, 2, 1], padding="SAME") + self.conv1_b
        x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)
        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling
        #logits = tf.matmul(x, self.fc_w) + self.fc_b
        logits = conv.matmul(x, self.fc_w) + self.fc_b

        return tf.nn.softmax(logits)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def __str__(self):
        lines = []
        lines.append("ResNet50_sparse2 Model Summary")
        lines.append("=" * 80)

        # Table Header
        lines.append(f"{'Layer Name':<30} {'Momentum Contribution':>25} {'Sparsity':>10} {'Active / Total':>15}")
        lines.append("-" * 80)

        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"
            sparsity_str = f"{sparsity:.1%}"

            lines.append(
                f"{tensor.name:<30} {momentum_pct:>25} {sparsity_str:>10} {f'{active:,} / {total:,}':>15}"
            )

        lines.append("-" * 80)

        # Model-level summary
        model_sparsity = self.model_sparsity()
        model_active = self.num_active_weights()
        model_total = self.num_weights()

        lines.append(f"{'Overall Model Sparsity:':<30} {model_sparsity:.1%}")
        lines.append(f"{'Total Active / Total Weights:':<30} {model_active:,} / {model_total:,}")

        lines.append("=" * 80)
        return "\n".join(lines)

    # TODO: sembra un getter
    def get_sparse_tensors(self):
        prunable = []
        def collect_sparse(module):
            # Include attributes of this module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, SparseTensor):
                    prunable.append(attr)
        collect_sparse(self)
        for submodule in self.submodules:
            collect_sparse(submodule)
        return prunable

    def prune(self, rho):
        prunable = self.sparse_tensors
        tot_pruned = 0
        for i, tensor in enumerate(prunable):
            pruned_i = tensor.prune(rho=rho)
            tot_pruned = tot_pruned + pruned_i
        return tot_pruned

    def update_relative_momentum_contributions(self, tensors):
        total = sum(t.mean_momentum for t in tensors)
        if total == 0:
            equal_share = 1.0 / len(tensors)
            for t in tensors:
                t.relative_momentum_contribution = equal_share
        else:
            for t in tensors:
                t.relative_momentum_contribution = t.mean_momentum / total


    def update_mean_momentums(self, optimizer):
        for sparse_tensor in self.sparse_tensors:
            w = sparse_tensor.values
            idx = optimizer._get_variable_index(w)
            momentum_tensor = optimizer._momentums[idx]
            sparse_tensor.mean_momentum = tf.reduce_mean(tf.abs(momentum_tensor))

    def regrow(self, to_regrow):
        # se vuoi capire perchè esiste regrowth_residual pensa al caso in cui to_regrow = 1

        if to_regrow > self.num_inactive_weights():
            raise ValueError(
                f"Cannot regrow {to_regrow} elements. "
                f"Only {self.num_inactive_weights()} positions available."
            )

        tot_regrown = 0
        remaining_to_allocate  = to_regrow
        active_tensors = self.sparse_tensors

        while remaining_to_allocate  != 0:
            self.update_relative_momentum_contributions(active_tensors)
            total_deficit = 0
            unsaturated_tensors = []
            for t in active_tensors:
                expected_regrow = int(remaining_to_allocate  * t.relative_momentum_contribution)

                actual_regrow = t.regrow(requested_growth=expected_regrow)
                tot_regrown = tot_regrown + actual_regrow
                deficit = expected_regrow - actual_regrow
                total_deficit = total_deficit + deficit
                if not t.is_saturated():
                    unsaturated_tensors.append(t)

            active_tensors = unsaturated_tensors
            remaining_to_allocate  = total_deficit

        residual = to_regrow - tot_regrown
        for t in active_tensors:
            regrown = t.regrow(residual)
            residual = residual-regrown
            if residual == 0:
                return

    def prune_and_regrow(self,rho,optimizer):
        print(f"rho: {rho}")
        self.update_mean_momentums(optimizer)
        self.reset_prune_and_regrow_stats()
        num_pruned = self.prune(rho)
        #print(self.prune_summary())
        self.regrow(num_pruned)
        #print(self.regrow_summary())

    def reset_prune_and_regrow_stats(self):
        for t in self.sparse_tensors:
            t.reset_prune_and_regrow_stats()

    def num_pruned(self):
        tot_pruned = 0
        for t in self.sparse_tensors:
            tot_pruned = tot_pruned + t.num_pruned
        return tot_pruned

    def num_regrown(self):
        tot_regrown = 0
        for t in self.sparse_tensors:
            tot_regrown = tot_regrown + t.num_regrown
        return tot_regrown

    def prune_summary(self):
        """Returns a detailed summary of pruning statistics for all sparse tensors.

        Returns:
            str: Formatted string containing pruning statistics for each tensor and global summary.
        """

        sep_length = 81

        lines = []
        lines.append("Pruning Summary")
        lines.append("=" * sep_length)

        # Table header
        lines.append(
            f"{'Layer Name':<15} {'Active/Total (b)':>22} {'Pruned':>8} {'Active/Total (a)':>22} {'Sparsity':>10}")
        lines.append("-" * sep_length)

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            pruned = tensor.num_pruned
            active_before_pruning = active + pruned
            sparsity = tensor.tensor_sparsity()

            lines.append(
                f"{tensor.name:<15} {f'{active_before_pruning:,}/{total:,}':>22} {pruned:>8} "
                f"{f'{active:,}/{total:,}':>22} {f'{sparsity:.1%}':>10}"
            )

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Pruned Weights: ", f"{self.num_pruned():,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def regrow_summary(self):
        """Returns a detailed summary of regrowth statistics for all sparse tensors.

        Returns:
            str: Formatted string containing regrowth statistics for each tensor and global summary.
        """
        sep_length = 88
        lines = []
        lines.append("Regrowth Summary")
        lines.append("=" * sep_length)

        # Table Header
        lines.append(
            f"{'Layer Name':<15} {'Momentum':>10} {'Active/Total (b)':>20} "
            f"{'Regrown':>8} {'Active/Total (a)':>20} {'Sparsity':>10}"
        )
        lines.append("-" * sep_length)

        total_regrown = 0

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active_after = tensor.num_active_weights()
            total = tensor.num_weights()
            regrown = tensor.num_regrown
            active_before = active_after - regrown
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"

            lines.append(
                f"{tensor.name:<15} {momentum_pct:>10} {f'{active_before:,}/{total:,}':>20} "
                f"{regrown:>8} {f'{active_after:,}/{total:,}':>20} {f'{sparsity:.1%}':>10}"
            )
            total_regrown += regrown

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Regrown Weights: ", f"{total_regrown:,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def model_sparsity(self):
        return self.num_inactive_weights()/self.num_weights()

    def num_active_weights(self):
        num_active_weights = 0
        for t in self.sparse_tensors:
            num_active_weights = num_active_weights + t.num_active_weights()
        return num_active_weights

    def num_weights(self):
        num_weights = 0
        for t in self.sparse_tensors:
            num_weights = num_weights + t.num_weights()
        return num_weights

    def num_inactive_weights(self):
        return self.num_weights() - self.num_active_weights()
'''

# checkpointing v2

class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, recompute=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            self.recompute = recompute

            self.w1 = SparseTensor([1, 1, in_channels, filters], sparsity, name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")

            self.w2 = SparseTensor([3, 3, filters, filters], sparsity, name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")

            self.w3 = SparseTensor([1, 1, filters, 4 * filters], sparsity, name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = SparseTensor([1, 1, in_channels, 4 * filters], sparsity, name="w_sc_M")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):
            def forward(x):
                shortcut = x
                if self.conv_shortcut:
                    shortcut = conv.sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                    shortcut = self.bn_sc(shortcut, training=training)

                x = conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1
                x = self.bn1(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2
                x = self.bn2(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3
                x = self.bn3(x, training=training)

                x = tf.nn.relu(x + shortcut)
                return x

            if self.recompute:
                forward = tf.recompute_grad(forward)

            return forward(x)

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, sparsity, recompute=False, name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(
                ResNet50_sparse2.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, recompute=recompute, name="block1")
            )
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_sparse2.ConvBlock(sparsity, 4 * filters, filters, stride=1, conv_shortcut=False, recompute=recompute, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:
                x = block(x, training=training)
            return x

    def __init__(self, sparsity, num_classes=8, recompute=False, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.recompute = recompute

        self.conv1_w = SparseTensor([7, 7, 3, 64], sparsity, name="conv1_w_M")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, recompute=recompute, name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, recompute=recompute, name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, recompute=recompute, name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, recompute=recompute, name="conv5")

        self.fc_w = SparseTensor([2048, num_classes], sparsity, name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

    def __call__(self, x, training=False):
        def stem(x):
            x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = self.pool(x)
            return x

        def head(x):
            x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling
            logits = conv.matmul(x, self.fc_w) + self.fc_b
            return tf.nn.softmax(logits)

        if self.recompute:
            x = tf.recompute_grad(stem)(x)
        else:
            x = stem(x)

        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)

        if self.recompute:
            return tf.recompute_grad(head)(x)
        else:
            return head(x)





# checkpointing v2 con prune & regrow
class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, recompute=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            self.recompute = recompute #recompute grad

            self.w1 = SparseTensor([1, 1, in_channels, filters], sparsity, name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")

            self.w2 = SparseTensor([3, 3, filters, filters], sparsity, name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")

            self.w3 = SparseTensor([1, 1, filters, 4 * filters], sparsity, name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = SparseTensor([1, 1, in_channels, 4 * filters], sparsity, name="w_sc_M")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):
            def forward(x):
                shortcut = x
                if self.conv_shortcut:
                    shortcut = conv.sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                    shortcut = self.bn_sc(shortcut, training=training)

                x = conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1
                x = self.bn1(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2
                x = self.bn2(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3
                x = self.bn3(x, training=training)

                x = tf.nn.relu(x + shortcut)
                return x

            if self.recompute:
                forward = tf.recompute_grad(forward)

            return forward(x)

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, sparsity, recompute=False, name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(
                ResNet50_sparse2.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, recompute=recompute, name="block1")
            )
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_sparse2.ConvBlock(sparsity, 4 * filters, filters, stride=1, conv_shortcut=False, recompute=recompute, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:
                x = block(x, training=training)
            return x

    def __init__(self, sparsity, num_classes=8, recompute=False, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.recompute = recompute

        self.conv1_w = SparseTensor([7, 7, 3, 64], sparsity, name="conv1_w_M")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, recompute=recompute, name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, recompute=recompute, name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, recompute=recompute, name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, recompute=recompute, name="conv5")

        self.fc_w = SparseTensor([2048, num_classes], sparsity, name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

        self.sparse_tensors = self.get_sparse_tensors()

    def __call__(self, x, training=False):
        def stem(x):
            x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = self.pool(x)
            return x

        def head(x):
            x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling
            logits = conv.matmul(x, self.fc_w) + self.fc_b
            return tf.nn.softmax(logits)

        if self.recompute:
            x = tf.recompute_grad(stem)(x)
        else:
            x = stem(x)

        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)

        if self.recompute:
            return tf.recompute_grad(head)(x)
        else:
            return head(x)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def __str__(self):
        lines = []
        lines.append("ResNet50_sparse2 Model Summary")
        lines.append("=" * 80)

        # Table Header
        lines.append(f"{'Layer Name':<30} {'Momentum Contribution':>25} {'Sparsity':>10} {'Active / Total':>15}")
        lines.append("-" * 80)

        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"
            sparsity_str = f"{sparsity:.1%}"

            lines.append(
                f"{tensor.name:<30} {momentum_pct:>25} {sparsity_str:>10} {f'{active:,} / {total:,}':>15}"
            )

        lines.append("-" * 80)

        # Model-level summary
        model_sparsity = self.model_sparsity()
        model_active = self.num_active_weights()
        model_total = self.num_weights()

        lines.append(f"{'Overall Model Sparsity:':<30} {model_sparsity:.1%}")
        lines.append(f"{'Total Active / Total Weights:':<30} {model_active:,} / {model_total:,}")

        lines.append("=" * 80)
        return "\n".join(lines)

    # TODO: sembra un getter
    def get_sparse_tensors(self):
        prunable = []
        def collect_sparse(module):
            # Include attributes of this module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, SparseTensor):
                    prunable.append(attr)
        collect_sparse(self)
        for submodule in self.submodules:
            collect_sparse(submodule)
        return prunable

    def prune(self, rho):
        prunable = self.sparse_tensors
        tot_pruned = 0
        for i, tensor in enumerate(prunable):
            pruned_i = tensor.prune(rho=rho)
            tot_pruned = tot_pruned + pruned_i
        return tot_pruned

    def update_relative_momentum_contributions(self, tensors):
        total = sum(t.mean_momentum for t in tensors)
        if total == 0:
            equal_share = 1.0 / len(tensors)
            for t in tensors:
                t.relative_momentum_contribution = equal_share
        else:
            for t in tensors:
                t.relative_momentum_contribution = t.mean_momentum / total


    def update_mean_momentums(self, optimizer):
        for sparse_tensor in self.sparse_tensors:
            w = sparse_tensor.values
            idx = optimizer._get_variable_index(w)
            momentum_tensor = optimizer._momentums[idx]
            sparse_tensor.mean_momentum = tf.reduce_mean(tf.abs(momentum_tensor))

    def regrow(self, to_regrow):

        # se vuoi capire perchè esiste regrowth_residual pensa al caso in cui to_regrow = 1
        if to_regrow > self.num_inactive_weights():
            raise ValueError(
                f"Cannot regrow {to_regrow} elements. "
                f"Only {self.num_inactive_weights()} positions available."
            )

        tot_regrown = 0
        remaining_to_allocate  = to_regrow
        active_tensors = self.sparse_tensors

        while remaining_to_allocate  != 0:
            self.update_relative_momentum_contributions(active_tensors)
            total_deficit = 0
            unsaturated_tensors = []
            for t in active_tensors:
                expected_regrow = int(remaining_to_allocate  * t.relative_momentum_contribution)

                actual_regrow = t.regrow(requested_growth=expected_regrow)
                tot_regrown = tot_regrown + actual_regrow
                deficit = expected_regrow - actual_regrow
                total_deficit = total_deficit + deficit
                if not t.is_saturated():
                    unsaturated_tensors.append(t)

            active_tensors = unsaturated_tensors
            remaining_to_allocate  = total_deficit

        residual = to_regrow - tot_regrown
        for t in active_tensors:
            regrown = t.regrow(residual)
            residual = residual-regrown
            if residual == 0:
                return

    def prune_and_regrow(self,rho,optimizer):
        print(f"rho: {rho}")
        self.update_mean_momentums(optimizer)
        self.reset_prune_and_regrow_stats()
        num_pruned = self.prune(rho)
        print(self.prune_summary())
        self.regrow(num_pruned)
        print(self.regrow_summary())

    def reset_prune_and_regrow_stats(self):
        for t in self.sparse_tensors:
            t.reset_prune_and_regrow_stats()

    def num_pruned(self):
        tot_pruned = 0
        for t in self.sparse_tensors:
            tot_pruned = tot_pruned + t.num_pruned
        return tot_pruned

    def num_regrown(self):
        tot_regrown = 0
        for t in self.sparse_tensors:
            tot_regrown = tot_regrown + t.num_regrown
        return tot_regrown

    def prune_summary(self):
        """Returns a detailed summary of pruning statistics for all sparse tensors.

        Returns:
            str: Formatted string containing pruning statistics for each tensor and global summary.
        """

        sep_length = 81

        lines = []
        lines.append("Pruning Summary")
        lines.append("=" * sep_length)

        # Table header
        lines.append(
            f"{'Layer Name':<15} {'Active/Total (b)':>22} {'Pruned':>8} {'Active/Total (a)':>22} {'Sparsity':>10}")
        lines.append("-" * sep_length)

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            pruned = tensor.num_pruned
            active_before_pruning = active + pruned
            sparsity = tensor.tensor_sparsity()

            lines.append(
                f"{tensor.name:<15} {f'{active_before_pruning:,}/{total:,}':>22} {pruned:>8} "
                f"{f'{active:,}/{total:,}':>22} {f'{sparsity:.1%}':>10}"
            )

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Pruned Weights: ", f"{self.num_pruned():,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def regrow_summary(self):
        """Returns a detailed summary of regrowth statistics for all sparse tensors.

        Returns:
            str: Formatted string containing regrowth statistics for each tensor and global summary.
        """
        sep_length = 88
        lines = []
        lines.append("Regrowth Summary")
        lines.append("=" * sep_length)

        # Table Header
        lines.append(
            f"{'Layer Name':<15} {'Momentum':>10} {'Active/Total (b)':>20} "
            f"{'Regrown':>8} {'Active/Total (a)':>20} {'Sparsity':>10}"
        )
        lines.append("-" * sep_length)

        total_regrown = 0

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active_after = tensor.num_active_weights()
            total = tensor.num_weights()
            regrown = tensor.num_regrown
            active_before = active_after - regrown
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"

            lines.append(
                f"{tensor.name:<15} {momentum_pct:>10} {f'{active_before:,}/{total:,}':>20} "
                f"{regrown:>8} {f'{active_after:,}/{total:,}':>20} {f'{sparsity:.1%}':>10}"
            )
            total_regrown += regrown

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Regrown Weights: ", f"{total_regrown:,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def model_sparsity(self):
        return self.num_inactive_weights()/self.num_weights()

    def num_active_weights(self):
        num_active_weights = 0
        for t in self.sparse_tensors:
            num_active_weights = num_active_weights + t.num_active_weights()
        return num_active_weights

    def num_weights(self):
        num_weights = 0
        for t in self.sparse_tensors:
            num_weights = num_weights + t.num_weights()
        return num_weights

    def num_inactive_weights(self):
        return self.num_weights() - self.num_active_weights()





#test
'''
#ResNet50_sparse & ResNet50_2(non sparse) ResNet50_sparse2(checkpointed v1/v2) devono dare gli stessi risultati -- servono solo per il debug -- ricorda di disabilitare oneNN custom ops
class ResNet50_sparse(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            initializer = tf.keras.initializers.HeNormal()

            # Conv layers
            #self.w1 = tf.Variable(initializer([1, 1, in_channels, filters]), name="w1")
            self.w1 = SparseTensor(v4.create_tensor_row_major(1, in_channels, filters),name="w1mio")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
            #-----------------------------------------------------------------
            #self.w2 = tf.Variable(initializer([3, 3, filters, filters]), name="w2")
            self.w2 = SparseTensor(v4.create_tensor_row_major(3, filters, filters),name="w2mio")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
            #--------------------------------------------------------------------
            #self.w3 = tf.Variable(initializer([1, 1, filters, 4 * filters]), name="w3")
            self.w3 = SparseTensor(v4.create_tensor_row_major(1, filters, 4*filters),name="w3mio")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                #self.w_sc = tf.Variable(initializer([1, 1, in_channels, 4 * filters]), name="w_sc")
                self.w_sc = SparseTensor(v4.create_tensor_row_major(1, in_channels, 4 * filters), name="w_scmio")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):
            shortcut = x
            if self.conv_shortcut:
                #shortcut = tf.nn.conv2d(x, self.w_sc, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.b_sc
                shortcut = conv.sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                shortcut = self.bn_sc(shortcut, training=training)

            #x = tf.nn.conv2d(x, self.w1, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.b1
            x = conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            #---------------------------------------------------
            #x = tf.nn.conv2d(x, self.w2, strides=[1, 1, 1, 1], padding="SAME") + self.b2
            x = conv.sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)
            #---------------------------------------------------

            #x = tf.nn.conv2d(x, self.w3, strides=[1, 1, 1, 1], padding="SAME") + self.b3
            x = conv.sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3
            x = self.bn3(x, training=training)
            x = tf.nn.relu(x + shortcut)
            return x

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, sparsity ,name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(ResNet50_sparse.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, name="block1"))
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_sparse.ConvBlock(sparsity,4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:
                x = block(x, training=training)
            return x


    def __init__(self, sparsity, num_classes=1000, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes

        initializer = tf.keras.initializers.HeNormal()

        # Initial conv
        #self.conv1_w = tf.Variable(initializer([7, 7, 3, 64]), name="conv1_w")
        self.conv1_w = SparseTensor(v4.create_tensor_row_major(7, 3, 64), name="conv1_wmio")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        # Residual stages
        self.stage2 = ResNet50_sparse.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity,name="conv2")
        self.stage3 = ResNet50_sparse.ResNetStack(256, 128, 4, stride1=2,  sparsity=sparsity,name="conv3")
        self.stage4 = ResNet50_sparse.ResNetStack(512, 256, 6, stride1=2,  sparsity=sparsity,name="conv4")
        self.stage5 = ResNet50_sparse.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, name="conv5")

        # Final FC layer
        #self.fc_w = tf.Variable(initializer([2048, num_classes]), name="fc_w")
        self.fc_w = SparseTensor([2048, num_classes],0 ,name="fc_wmio")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

    def __call__(self, x, training=False):
        #tf.io.write_file('sp.bytes', tf.io.serialize_tensor(x))
        #exit()
        #x = tf.nn.conv2d(x, self.conv1_w, strides=[1, 2, 2, 1], padding="SAME") + self.conv1_b
        x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b



        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)

        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling

        #logits = tf.matmul(x, self.fc_w) + self.fc_b
        logits = conv.matmul(x, self.fc_w) + self.fc_b

        return tf.nn.softmax(logits)
class ResNet50_2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            initializer = tf.keras.initializers.HeNormal()

            # Conv layers
            #self.w1 = tf.Variable(initializer([1, 1, in_channels, filters]), name="w1")
            self.w1 = tf.Variable(v4.create_tensor_row_major(1, in_channels, filters),name="w1mio")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
#           ------------------------------------------------------------------------------------
            #self.w2 = tf.Variable(initializer([3, 3, filters, filters]), name="w2")
            self.w2 = tf.Variable(v4.create_tensor_row_major(3, filters, filters),name="w2mio")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
#            ------------------------------------------------------------------------------------
            #self.w3 = tf.Variable(initializer([1, 1, filters, 4 * filters]), name="w3")
            self.w3 = tf.Variable(v4.create_tensor_row_major(1, filters,4* filters),name="w3mio")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                #self.w_sc = tf.Variable(initializer([1, 1, in_channels, 4 * filters]), name="w_sc")
                self.w_sc = tf.Variable(v4.create_tensor_row_major(1, in_channels, 4 * filters), name="w_scmio")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):
            shortcut = x
            if self.conv_shortcut:
                shortcut = tf.nn.conv2d(x, self.w_sc, strides=[1, self.stride, self.stride, 1],padding="SAME") + self.b_sc
                shortcut = self.bn_sc(shortcut, training=training)

            x = tf.nn.conv2d(x, self.w1, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.b1
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)

            x = tf.nn.conv2d(x, self.w2, strides=[1, 1, 1, 1], padding="SAME") + self.b2
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)

            x = tf.nn.conv2d(x, self.w3, strides=[1, 1, 1, 1], padding="SAME") + self.b3
            x = self.bn3(x, training=training)

            x = tf.nn.relu(x + shortcut)
            return x

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(ResNet50_2.ConvBlock(in_channels, filters, stride=stride1, conv_shortcut=True, name="block1"))
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_2.ConvBlock(4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:
                x = block(x, training=training)
            return x


    def __init__(self, num_classes=1000, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes

        initializer = tf.keras.initializers.HeNormal()

        # Initial conv
        #self.conv1_w = tf.Variable(initializer([7, 7, 3, 64]), name="conv1_w")
        self.conv1_w = tf.Variable(v4.create_tensor_row_major(7, 3, 64), name="conv1_wmio")

        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        # Residual stages
        self.stage2 = ResNet50_2.ResNetStack(64, 64, 3, stride1=1, name="conv2")
        self.stage3 = ResNet50_2.ResNetStack(256, 128, 4, stride1=2, name="conv3")
        self.stage4 = ResNet50_2.ResNetStack(512, 256, 6, stride1=2, name="conv4")
        self.stage5 = ResNet50_2.ResNetStack(1024, 512, 3, stride1=2, name="conv5")

        # Final FC layer
        #self.fc_w = tf.Variable(initializer([2048, num_classes]), name="fc_w")
        self.fc_w =  tf.Variable(SparseTensor([2048, num_classes],0 ,name="fc_wmio").to_tf_dense(), name="fc_wmio")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

    def __call__(self, x, training=False):
        #tf.io.write_file('de.bytes', tf.io.serialize_tensor(x))
        #exit()
        x = tf.nn.conv2d(x, self.conv1_w, strides=[1, 2, 2, 1], padding="SAME") + self.conv1_b


        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)

        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling

        logits = tf.matmul(x, self.fc_w) + self.fc_b
        return tf.nn.softmax(logits)
#v1 (senza il flag & alcuni layer non sono ricalcolati)
class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut

            self.w1 = SparseTensor(v4.create_tensor_row_major(1, in_channels, filters),name="w1mio")
            #self.w1 = funzioni.SparseTensor([1, 1, in_channels, filters], sparsity, name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")

            self.w2 = SparseTensor(v4.create_tensor_row_major(3, filters, filters),name="w2mio")
            #self.w2 = funzioni.SparseTensor([3, 3, filters, filters], sparsity, name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")

            self.w3 = SparseTensor(v4.create_tensor_row_major(1, filters, 4*filters),name="w3mio")
            #self.w3 = funzioni.SparseTensor([1, 1, filters, 4 * filters], sparsity, name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = SparseTensor(v4.create_tensor_row_major(1, in_channels, 4 * filters), name="w_scmio")

                #self.w_sc = funzioni.SparseTensor([1, 1, in_channels, 4 * filters], sparsity, name="w_sc_M")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):

            @tf.recompute_grad
            def forward(x):
                shortcut = x
                if self.conv_shortcut:
                    shortcut = conv.sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                    shortcut = self.bn_sc(shortcut, training=training)

                x = conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1
                x = self.bn1(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2
                x = self.bn2(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3
                x = self.bn3(x, training=training)

                x = tf.nn.relu(x + shortcut)
                return x

            return forward(x)

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, sparsity, name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(
                ResNet50_sparse2.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, name="block1")
            )
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_sparse2.ConvBlock(sparsity, 4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:
                x = block(x, training=training)
            return x

    def __init__(self, sparsity, num_classes=1000, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes

        #self.conv1_w = funzioni.SparseTensor([7, 7, 3, 64], sparsity, name="conv1_w_M")
        self.conv1_w = SparseTensor(v4.create_tensor_row_major(7, 3, 64), name="conv1_wmio")

        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, name="conv5")

        self.fc_w = SparseTensor([2048, num_classes],0 ,name="fc_wmio")
        #self.fc_w = funzioni.SparseTensor([2048, num_classes], sparsity, name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

    def __call__(self, x, training=False):
        x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)

        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling

        logits = conv.matmul(x, self.fc_w) + self.fc_b
        return tf.nn.softmax(logits)
#v2 (con flag e ricalcolo tutti layer)
class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, recompute=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            self.recompute = recompute

            self.w1 = SparseTensor(v4.create_tensor_row_major(1, in_channels, filters),name="w1mio")
            #self.w1 = SparseTensor([1, 1, in_channels, filters], sparsity, name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")

            self.w2 = SparseTensor(v4.create_tensor_row_major(3, filters, filters),name="w2mio")
            #self.w2 = SparseTensor([3, 3, filters, filters], sparsity, name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")

            self.w3 = SparseTensor(v4.create_tensor_row_major(1, filters, 4*filters),name="w3mio")
            #self.w3 = SparseTensor([1, 1, filters, 4 * filters], sparsity, name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = SparseTensor(v4.create_tensor_row_major(1, in_channels, 4 * filters), name="w_scmio")
                #self.w_sc = SparseTensor([1, 1, in_channels, 4 * filters], sparsity, name="w_sc_M")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):
            def forward(x):
                shortcut = x
                if self.conv_shortcut:
                    shortcut = conv.sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                    shortcut = self.bn_sc(shortcut, training=training)

                x = conv.sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1
                x = self.bn1(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2
                x = self.bn2(x, training=training)
                x = tf.nn.relu(x)

                x = conv.sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3
                x = self.bn3(x, training=training)

                x = tf.nn.relu(x + shortcut)
                return x

            if self.recompute:
                forward = tf.recompute_grad(forward)

            return forward(x)

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, sparsity, recompute=False, name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(
                ResNet50_sparse2.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, recompute=recompute, name="block1")
            )
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_sparse2.ConvBlock(sparsity, 4 * filters, filters, stride=1, conv_shortcut=False, recompute=recompute, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:
                x = block(x, training=training)
            return x

    def __init__(self, sparsity, num_classes=8, recompute=False, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.recompute = recompute

        self.conv1_w = SparseTensor(v4.create_tensor_row_major(7, 3, 64), name="conv1_wmio")
        #self.conv1_w = SparseTensor([7, 7, 3, 64], sparsity, name="conv1_w_M")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, recompute=recompute, name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, recompute=recompute, name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, recompute=recompute, name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, recompute=recompute, name="conv5")

        self.fc_w = SparseTensor([2048, num_classes],0 ,name="fc_wmio")
        #self.fc_w = SparseTensor([2048, num_classes], sparsity, name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

    def __call__(self, x, training=False):
        def stem(x):
            x = conv.sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = self.pool(x)
            return x

        def head(x):
            x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling
            logits = conv.matmul(x, self.fc_w) + self.fc_b
            return tf.nn.softmax(logits)

        if self.recompute:
            x = tf.recompute_grad(stem)(x)
        else:
            x = stem(x)

        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)

        if self.recompute:
            return tf.recompute_grad(head)(x)
        else:
            return head(x)
'''

#TODO: x = tf.nn.conv2d(x, self.w1, strides=..., padding=...) + self.b1 vs x = tf.nn.bias_add(tf.nn.conv2d(x, self.w1, strides=..., padding=...), self.b1)
#TODO: dropput
#TODO: SparseCategoricalCrossentropy(from_logits=True) cos'è?
#TODO: You're recreating BatchNormalization layers directly in the block constructors — that's fine, but remember: They must be reused correctly during training and inference. You're doing this right — just keep this in mind when saving/loading
# resnet original
'''
class ResNet50_original(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            initializer = tf.keras.initializers.HeNormal()

            # Conv layers
            self.w1 = tf.Variable(initializer([1, 1, in_channels, filters]), name="w1")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")

            self.w2 = tf.Variable(initializer([3, 3, filters, filters]), name="w2")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")

            self.w3 = tf.Variable(initializer([1, 1, filters, 4 * filters]), name="w3")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = tf.Variable(initializer([1, 1, in_channels, 4 * filters]), name="w_sc")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = tf.keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):
            shortcut = x
            if self.conv_shortcut:
                shortcut = tf.nn.conv2d(x, self.w_sc, strides=[1, self.stride, self.stride, 1],
                                        padding="SAME") + self.b_sc
                shortcut = self.bn_sc(shortcut, training=training)

            x = tf.nn.conv2d(x, self.w1, strides=[1, self.stride, self.stride, 1], padding="SAME") + self.b1
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)

            x = tf.nn.conv2d(x, self.w2, strides=[1, 1, 1, 1], padding="SAME") + self.b2
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)

            x = tf.nn.conv2d(x, self.w3, strides=[1, 1, 1, 1], padding="SAME") + self.b3
            x = self.bn3(x, training=training)

            x = tf.nn.relu(x + shortcut)
            return x

    class ResNetStack(tf.Module):
        def __init__(self, in_channels, filters, blocks, stride1, name=None):
            super().__init__(name=name)
            self.blocks = []

            self.blocks.append(ResNet50_original.ConvBlock(in_channels, filters, stride=stride1, conv_shortcut=True, name="block1"))
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50_original.ConvBlock(4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
                )

        def __call__(self, x, training=False):
            for block in self.blocks:
                x = block(x, training=training)
            return x


    def __init__(self, num_classes=1000, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes

        initializer = tf.keras.initializers.HeNormal()

        # Initial conv
        self.conv1_w = tf.Variable(initializer([7, 7, 3, 64]), name="conv1_w")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        # Residual stages
        self.stage2 = ResNet50_original.ResNetStack(64, 64, 3, stride1=1, name="conv2")
        self.stage3 = ResNet50_original.ResNetStack(256, 128, 4, stride1=2, name="conv3")
        self.stage4 = ResNet50_original.ResNetStack(512, 256, 6, stride1=2, name="conv4")
        self.stage5 = ResNet50_original.ResNetStack(1024, 512, 3, stride1=2, name="conv5")

        # Final FC layer
        self.fc_w = tf.Variable(initializer([2048, num_classes]), name="fc_w")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

    def __call__(self, x, training=False):
        x = tf.nn.conv2d(x, self.conv1_w, strides=[1, 2, 2, 1], padding="SAME") + self.conv1_b
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)

        x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling

        logits = tf.matmul(x, self.fc_w) + self.fc_b
        return tf.nn.softmax(logits)
'''
# resnet Keras original
'''
def ResNet50_keras(input_shape=(224, 224, 3), num_classes=1000):
    def resnet_stack(x, filters, blocks, stride1):
        # stride1 = 2 tranne il primo stack
        x = conv_block(x, filters, stride = stride1, shortcut_conv=True)
        for _ in range(1, blocks):
            x = conv_block(x, filters, stride = 1, shortcut_conv=False)
        return x

    def conv_block(x, filters, stride, shortcut_conv):
        # secondo stack
        # filters = 128
        # stride = 2, shortcut_conv = True se è il primo blocco, stride = 1, shortcut_conv = False, altrimenti
        # (256 channels se primo blocco, 512 altrimenti)
        # (56x56 se primo blocco, 28x28 altrimenti)
        shortcut = x

        # entra qui solo se è il primo blocco dello stack perchè dobbiamo dimezzare la risoluzione portare channels da 256 -> 512
        # se è il primo blocco, allora stride = 2, x = (N,56,56,256) -> shortcut = (N,28,28,512)
        if shortcut_conv:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # questa è la prima operazione ufficiale
        # se primo blocco, stride = 2, (N,56,56,256) -> (N,28,28,128)
        # se non primo blocco, stride = 1, (N,28,28,512) -> (N,28,28,128)
        x = layers.Conv2D(filters, 1, strides=stride)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # stride = 1, (N,28,28,128) -> (N,28,28,128) in ogni caso
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # stride = 1, (N,28,28,128) -> (N,28,28,512) in ogni caso
        x = layers.Conv2D(4 * filters, 1)(x)
        x = layers.BatchNormalization()(x)

        # (N,28,28,512) + (N,28,28,512)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    inputs = tf.keras.Input(shape=input_shape)
    # il secondo parametro è kernel_size
    x = layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # the downsampling (halving of spatial dimensions) is due to the strides=2, not the pool_size
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = resnet_stack(x, 64, 3, stride1=1)

    x = resnet_stack(x, 128, 4,stride1=2)
    x = resnet_stack(x, 256, 6, stride1=2)
    x = resnet_stack(x, 512, 3,stride1=2)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
'''

# mobile net base
'''
def MobileNet_keras(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, classes=8):
    def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
        filters = int(filters * alpha)
        #filters: output_ch
        x = Conv2D(filters, kernel, padding="same", use_bias=False, strides=strides)(inputs)
        x = BatchNormalization()(x)
        return ReLU(6.)(x)

    # H x W x C_in -> H1 x W1 x pointwise_filters
    # praticamente è Conv2D(pointwise_filters,strides, padding = "SAME")
    def _depthwise_conv_block(inputs, pointwise_filters, alpha, depth_multiplier=1, strides=(1, 1)):
        pointwise_filters = int(pointwise_filters * alpha)
        # HxWxC_in -> H1xH2xC_in
        x = DepthwiseConv2D((3, 3), padding="same", depth_multiplier=depth_multiplier,
                            strides=strides, use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        # H1 x H2 x C_in -> H1 x H2 x pointwise_filters
        x = Conv2D(pointwise_filters, (1, 1), padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        return ReLU(6.)(x)

    img_input = Input(shape=input_shape)

    # 224x224x3 -> 112x112x32
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))

    # pointwise_filters: out_chan
    # se strides = (1,1) significa che non riduco H,W

    # 112x112x32 -> 112x112x64 (rosa)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier)

    # 112x112x64 -> 56x56x128 (blu)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2))

    # 56x56x128 -> 56x56x128 (arancione)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier)

    # 56x56x128 -> 28x28x256 (viola)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2))

    # 28x28x256 -> 28x28x256 (verde)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier)

    # 28x28x256 -> 14x14x512 (azzurro)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2))

    for _ in range(5):
        # 14x14x512 -> 14x14x512
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier)

    # 14x14x512 -> 7x7x1024
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2))

    # 7x7x1024 -> 7x7x1024
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier)

    x = GlobalAveragePooling2D()(x)
    #x = Dropout(dropout)(x)

    x = Dense(classes, activation="softmax")(x)

    return Model(img_input, x)
class MobileNetTF(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, name=None):
            super().__init__(name=name)
            self.stride = stride
            # self.kernel_size = kernel_size
            self.conv_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, out_channels)),
                                            name="conv")

            self.bn = BatchNormalization()

        def __call__(self, x, training=False):
            x = tf.nn.conv2d(x, self.conv_weights, strides=self.stride, padding="SAME")
            x = self.bn(x, training=training)
            return tf.nn.relu6(x)

    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]

            self.dw_weights = tf.Variable(
                tf.initializers.GlorotUniform()(
                    shape=(3, 3, in_channels, 1)  # depth_multiplier=1
                ),
                trainable=True, name="dw_weights"
            )
            self.pw_weights = tf.Variable(
                tf.initializers.GlorotUniform()(
                    shape=(1, 1, in_channels, out_channels)
                ),
                trainable=True, name="pw_weights"
            )
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

        def __call__(self, x, training=False):

            x = tf.nn.depthwise_conv2d(x, self.dw_weights, strides=self.strides, padding="SAME")
            x = self.bn1(x, training=training)
            x = tf.nn.relu6(x)

            x = tf.nn.conv2d(x, self.pw_weights, strides=1, padding="SAME")
            x = self.bn2(x, training=training)
            return tf.nn.relu6(x)
    def __init__(self, classes=8, name=None):
        super().__init__(name=name)
        self.blocks = []
        self.classes = classes

        # Define blocks with explicit channel sizes
        self.blocks.append(MobileNetTF.ConvBlock(3, 32, stride=2, name="conv1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(32, 64,stride=1, name="dw1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(64, 128, stride=2, name="dw2"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 128,stride = 1, name="dw3"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 256, stride=2, name="dw4"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 256,stride = 1, name="dw5"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 512, stride=2, name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 512, stride = 1,name=f"dw7_{i}"))

        self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 1024, stride=2, name="dw8"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(1024, 1024,stride = 1 ,name="dw9"))

        self.global_pool = GlobalAveragePooling2D()
        self.dense_weights = tf.Variable(
            tf.initializers.GlorotUniform()(shape=(1024, classes)),
            trainable=True, name="dense_weights"
        )
        self.dense_bias = tf.Variable(tf.zeros([classes]), trainable=True, name="dense_bias")
    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        x = self.global_pool(x)
        x = tf.matmul(x, self.dense_weights) + self.dense_bias
        return tf.nn.softmax(x)
'''

#test MobileNetTF (std) & MobileNetTF_Sparse MobileNetTF (check, p%r) devono dare gli stessi risultati
'''
class MobileNetTF(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, name=None):
            super().__init__(name=name)
            self.stride = stride
            # self.kernel_size = kernel_size
            #self.conv_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, out_channels)),name="conv")
            self.conv_weights = tf.Variable(tf.ones([3,3,in_channels, out_channels],name="conv"))

            self.bn = BatchNormalization()

        def __call__(self, x, training=False):
            x = tf.nn.conv2d(x, self.conv_weights, strides=self.stride, padding="SAME")
            x = self.bn(x, training=training)
            return tf.nn.relu6(x)

    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]

            #self.dw_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, 1),trainable=True, name="dw_weights"))
            self.dw_weights = tf.Variable(tf.ones([3,3,in_channels,1]), name="dw_weights")
            #self.pw_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, 1, in_channels, out_channels)),trainable=True, name="pw_weights")
            self.pw_weights = tf.Variable( tf.ones([1,1,in_channels,out_channels]) ,name="pw_weights")
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

        def __call__(self, x, training=False):

            x = tf.nn.depthwise_conv2d(x, self.dw_weights, strides=self.strides, padding="SAME")
            x = self.bn1(x, training=training)
            x = tf.nn.relu6(x)

            x = tf.nn.conv2d(x, self.pw_weights, strides=1, padding="SAME")
            x = self.bn2(x, training=training)
            return tf.nn.relu6(x)
    def __init__(self, num_classes=8, name=None):
        super().__init__(name=name)
        self.blocks = []

        # Define blocks with explicit channel sizes
        self.blocks.append(MobileNetTF.ConvBlock(3, 32, stride=2, name="conv1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(32, 64,stride=1, name="dw1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(64, 128, stride=2, name="dw2"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 128,stride = 1, name="dw3"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 256, stride=2, name="dw4"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 256,stride = 1, name="dw5"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 512, stride=2, name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 512, stride = 1,name=f"dw7_{i}"))

        self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 1024, stride=2, name="dw8"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(1024, 1024,stride = 1 ,name="dw9"))

        self.global_pool = GlobalAveragePooling2D()
        #self.dense_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1024, classes)),trainable=True, name="dense_weights")
        self.dense_weights = tf.Variable(tf.ones([1024,num_classes], name="dense_weights"))
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")
    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        x = self.global_pool(x)
        x = tf.matmul(x, self.dense_weights) + self.dense_bias
        return tf.nn.softmax(x)
'''
'''
class MobileNetTF_Sparse(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, name=None):
            super().__init__(name=name)
            self.stride = stride
            #self.conv_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, out_channels)),name="conv")
            self.conv_weights = SparseTensor(tf.ones([3,3,in_channels, out_channels], name="conv"))
            self.bn = BatchNormalization()

        def __call__(self, x, training=False):
            #x = tf.nn.conv2d(x, self.conv_weights, strides=self.stride, padding="SAME")
            x = conv.sparse_to_dense_conv2d(x,self.conv_weights,self.stride,padding="SAME")
            x = self.bn(x, training=training)
            return tf.nn.relu6(x)
    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]

            #self.dw_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, 1)), trainable=True, name="dw_weights")
            self.dw_weights = SparseTensor(tf.ones([3,3,in_channels,1]))

            #self.pw_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, 1, in_channels, out_channels)),trainable=True, name="pw_weights")
            self.pw_weights = SparseTensor(tf.ones([1,1,in_channels,out_channels]))
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

        def __call__(self, x, training=False):
            #x = tf.nn.depthwise_conv2d(x, self.dw_weights, strides=self.strides, padding="SAME")
            x = conv.sparse_to_dense_depthwise_conv2d(x,self.dw_weights,self.strides,padding="SAME")
            x = self.bn1(x, training=training)
            x = tf.nn.relu6(x)

            #x = tf.nn.conv2d(x, self.pw_weights, strides=1, padding="SAME")
            x = conv.sparse_to_dense_conv2d(x,self.pw_weights,stride=1,padding="SAME")
            x = self.bn2(x, training=training)
            return tf.nn.relu6(x)

    def __init__(self, num_classes=8, name=None):
        super().__init__(name=name)
        self.blocks = []

        # Define blocks with explicit channel sizes
        self.blocks.append(MobileNetTF.ConvBlock(3, 32, stride=2, name="conv1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(32, 64,stride=1, name="dw1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(64, 128, stride=2, name="dw2"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 128,stride = 1, name="dw3"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 256, stride=2, name="dw4"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 256,stride = 1, name="dw5"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 512, stride=2, name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 512, stride = 1,name=f"dw7_{i}"))

        self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 1024, stride=2, name="dw8"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(1024, 1024,stride = 1 ,name="dw9"))

        self.global_pool = GlobalAveragePooling2D()

        #self.dense_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1024, num_classes)),trainable=True, name="dense_weights" )
        self.dense_weights = SparseTensor(tf.ones([1024,num_classes]),name="fc_wmio")
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")
    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        x = self.global_pool(x)
        #x = tf.matmul(x, self.dense_weights) + self.dense_bias
        x = conv.matmul(x,self.dense_weights) + self.dense_bias
        return tf.nn.softmax(x)
'''
'''
class MobileNetTF(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.recompute = recompute_gradient
            #self.conv_weights = SparseTensor([3, 3, in_channels, out_channels], sparsity, name="conv")
            self.conv_weights = SparseTensor(tf.ones([3,3,in_channels, out_channels], name="conv"))

            self.bn = BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = conv.sparse_to_dense_conv2d(x, self.conv_weights, self.stride, padding="SAME")
                x = self.bn(x, training=training)
                return tf.nn.relu6(x)

            if self.recompute:
                return tf.recompute_grad(forward)(x)
            else:
                return forward(x)

    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]
            self.recompute = recompute_gradient
            #self.dw_weights = SparseTensor([3, 3, in_channels, 1], sparsity, name="dw_weights")
            self.dw_weights = SparseTensor(tf.ones([3,3,in_channels,1]))

            #self.pw_weights = SparseTensor([1, 1, in_channels, out_channels], sparsity, name="pw_weights")
            self.pw_weights = SparseTensor(tf.ones([1,1,in_channels,out_channels]))

            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = conv.sparse_to_dense_depthwise_conv2d(x, self.dw_weights, self.strides, padding="SAME")
                x = self.bn1(x, training=training)
                x = tf.nn.relu6(x)

                x = conv.sparse_to_dense_conv2d(x, self.pw_weights, stride=1, padding="SAME")
                x = self.bn2(x, training=training)
                return tf.nn.relu6(x)

            if self.recompute:
                return tf.recompute_grad(forward)(x)
            else:
                return forward(x)

    def __init__(self, sparsity, num_classes=8, recompute_gradient=False, name=None):
        super().__init__(name=name)
        self.recompute = recompute_gradient
        self.blocks = []

        # Define all blocks with recompute flag
        self.blocks.append(MobileNetTF.ConvBlock(3, 32, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="conv1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(32, 64, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(64, 128, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw2"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 128, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw3"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 256, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw4"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 256, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw5"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 512, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 512, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name=f"dw7_{i}"))

        self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 1024, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw8"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(1024, 1024, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw9"))

        self.global_pool = GlobalAveragePooling2D()
        #self.dense_weights = SparseTensor([1024, num_classes], sparsity, name="dense")
        self.dense_weights = SparseTensor(tf.ones([1024,num_classes]),name="fc_wmio")

        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")

        self.sparse_tensors = self.get_sparse_tensors()

    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)

        def head(x):
            x = self.global_pool(x)
            x = conv.matmul(x, self.dense_weights) + self.dense_bias
            return tf.nn.softmax(x)

        if self.recompute:
            return tf.recompute_grad(head)(x)
        else:
            return head(x)

    def __str__(self):
        lines = []
        lines.append("ResNet50_sparse2 Model Summary")
        lines.append("=" * 80)

        # Table Header
        lines.append(f"{'Layer Name':<30} {'Momentum Contribution':>25} {'Sparsity':>10} {'Active / Total':>15}")
        lines.append("-" * 80)

        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"
            sparsity_str = f"{sparsity:.1%}"

            lines.append(
                f"{tensor.name:<30} {momentum_pct:>25} {sparsity_str:>10} {f'{active:,} / {total:,}':>15}"
            )

        lines.append("-" * 80)

        # Model-level summary
        model_sparsity = self.model_sparsity()
        model_active = self.num_active_weights()
        model_total = self.num_weights()

        lines.append(f"{'Overall Model Sparsity:':<30} {model_sparsity:.1%}")
        lines.append(f"{'Total Active / Total Weights:':<30} {model_active:,} / {model_total:,}")

        lines.append("=" * 80)
        return "\n".join(lines)

    # TODO: sembra un getter
    def get_sparse_tensors(self):
        prunable = []
        def collect_sparse(module):
            # Include attributes of this module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, SparseTensor):
                    prunable.append(attr)
        collect_sparse(self)
        for submodule in self.submodules:
            collect_sparse(submodule)
        return prunable

    def prune(self, rho):
        prunable = self.sparse_tensors
        tot_pruned = 0
        for i, tensor in enumerate(prunable):
            pruned_i = tensor.prune(rho=rho)
            tot_pruned = tot_pruned + pruned_i
        return tot_pruned

    def update_relative_momentum_contributions(self, tensors):
        total = sum(t.mean_momentum for t in tensors)
        if total == 0:
            equal_share = 1.0 / len(tensors)
            for t in tensors:
                t.relative_momentum_contribution = equal_share
        else:
            for t in tensors:
                t.relative_momentum_contribution = t.mean_momentum / total

    def update_mean_momentums(self, optimizer):
        for sparse_tensor in self.sparse_tensors:
            w = sparse_tensor.values
            idx = optimizer._get_variable_index(w)
            momentum_tensor = optimizer._momentums[idx]
            sparse_tensor.mean_momentum = tf.reduce_mean(tf.abs(momentum_tensor))

    def regrow(self, to_regrow):

        # se vuoi capire perchè esiste regrowth_residual pensa al caso in cui to_regrow = 1
        if to_regrow > self.num_inactive_weights():
            raise ValueError(
                f"Cannot regrow {to_regrow} elements. "
                f"Only {self.num_inactive_weights()} positions available."
            )

        tot_regrown = 0
        remaining_to_allocate  = to_regrow
        active_tensors = self.sparse_tensors

        while remaining_to_allocate  != 0:
            self.update_relative_momentum_contributions(active_tensors)
            total_deficit = 0
            unsaturated_tensors = []
            for t in active_tensors:
                expected_regrow = int(remaining_to_allocate  * t.relative_momentum_contribution)

                actual_regrow = t.regrow(requested_growth=expected_regrow)
                tot_regrown = tot_regrown + actual_regrow
                deficit = expected_regrow - actual_regrow
                total_deficit = total_deficit + deficit
                if not t.is_saturated():
                    unsaturated_tensors.append(t)

            active_tensors = unsaturated_tensors
            remaining_to_allocate  = total_deficit

        residual = to_regrow - tot_regrown
        for t in active_tensors:
            regrown = t.regrow(residual)
            residual = residual-regrown
            if residual == 0:
                return

    def prune_and_regrow(self,rho,optimizer):
        print(f"rho: {rho}")
        self.update_mean_momentums(optimizer)
        self.reset_prune_and_regrow_stats()
        num_pruned = self.prune(rho)
        print(self.prune_summary())
        self.regrow(num_pruned)
        print(self.regrow_summary())

    def reset_prune_and_regrow_stats(self):
        for t in self.sparse_tensors:
            t.reset_prune_and_regrow_stats()

    def num_pruned(self):
        tot_pruned = 0
        for t in self.sparse_tensors:
            tot_pruned = tot_pruned + t.num_pruned
        return tot_pruned

    def num_regrown(self):
        tot_regrown = 0
        for t in self.sparse_tensors:
            tot_regrown = tot_regrown + t.num_regrown
        return tot_regrown

    def prune_summary(self):
        """Returns a detailed summary of pruning statistics for all sparse tensors.

        Returns:
            str: Formatted string containing pruning statistics for each tensor and global summary.
        """

        sep_length = 81

        lines = []
        lines.append("Pruning Summary")
        lines.append("=" * sep_length)

        # Table header
        lines.append(
            f"{'Layer Name':<15} {'Active/Total (b)':>22} {'Pruned':>8} {'Active/Total (a)':>22} {'Sparsity':>10}")
        lines.append("-" * sep_length)

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            pruned = tensor.num_pruned
            active_before_pruning = active + pruned
            sparsity = tensor.tensor_sparsity()

            lines.append(
                f"{tensor.name:<15} {f'{active_before_pruning:,}/{total:,}':>22} {pruned:>8} "
                f"{f'{active:,}/{total:,}':>22} {f'{sparsity:.1%}':>10}"
            )

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Pruned Weights: ", f"{self.num_pruned():,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def regrow_summary(self):
        """Returns a detailed summary of regrowth statistics for all sparse tensors.

        Returns:
            str: Formatted string containing regrowth statistics for each tensor and global summary.
        """
        sep_length = 88
        lines = []
        lines.append("Regrowth Summary")
        lines.append("=" * sep_length)

        # Table Header
        lines.append(
            f"{'Layer Name':<15} {'Momentum':>10} {'Active/Total (b)':>20} "
            f"{'Regrown':>8} {'Active/Total (a)':>20} {'Sparsity':>10}"
        )
        lines.append("-" * sep_length)

        total_regrown = 0

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active_after = tensor.num_active_weights()
            total = tensor.num_weights()
            regrown = tensor.num_regrown
            active_before = active_after - regrown
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"

            lines.append(
                f"{tensor.name:<15} {momentum_pct:>10} {f'{active_before:,}/{total:,}':>20} "
                f"{regrown:>8} {f'{active_after:,}/{total:,}':>20} {f'{sparsity:.1%}':>10}"
            )
            total_regrown += regrown

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Regrown Weights: ", f"{total_regrown:,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def model_sparsity(self):
        return self.num_inactive_weights()/self.num_weights()

    def num_active_weights(self):
        num_active_weights = 0
        for t in self.sparse_tensors:
            num_active_weights = num_active_weights + t.num_active_weights()
        return num_active_weights

    def num_weights(self):
        num_weights = 0
        for t in self.sparse_tensors:
            num_weights = num_weights + t.num_weights()
        return num_weights

    def num_inactive_weights(self):
        return self.num_weights() - self.num_active_weights()
'''


# mobile net prune and regrow
'''
class MobileNetTF(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity ,name=None):
            super().__init__(name=name)
            self.stride = stride
            #self.conv_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, out_channels)),name="conv")
            self.conv_weights = SparseTensor([3,3,in_channels,out_channels],sparsity, name="conv")
            self.bn = BatchNormalization()

        def __call__(self, x, training=False):
            #x = tf.nn.conv2d(x, self.conv_weights, strides=self.stride, padding="SAME")
            x = conv.sparse_to_dense_conv2d(x,self.conv_weights,self.stride,padding="SAME")
            x = self.bn(x, training=training)
            return tf.nn.relu6(x)
    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity ,name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]

            #self.dw_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, 1)), trainable=True, name="dw_weights")
            self.dw_weights = SparseTensor([3,3,in_channels,1],sparsity, name="dw_weights")

            #self.pw_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, 1, in_channels, out_channels)),trainable=True, name="pw_weights")
            self.pw_weights = SparseTensor([1,1,in_channels,out_channels],sparsity, name="pw_weights")
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

        def __call__(self, x, training=False):
            #x = tf.nn.depthwise_conv2d(x, self.dw_weights, strides=self.strides, padding="SAME")
            x = conv.sparse_to_dense_depthwise_conv2d(x,self.dw_weights,self.strides,padding="SAME")
            x = self.bn1(x, training=training)
            x = tf.nn.relu6(x)

            #x = tf.nn.conv2d(x, self.pw_weights, strides=1, padding="SAME")
            x = conv.sparse_to_dense_conv2d(x,self.pw_weights,stride=1,padding="SAME")
            x = self.bn2(x, training=training)
            return tf.nn.relu6(x)

    def __init__(self,sparsity, num_classes=8, name=None):
        super().__init__(name=name)
        self.blocks = []

        # Define blocks with explicit channel sizes
        self.blocks.append(MobileNetTF.ConvBlock(3, 32, stride=2, sparsity = sparsity, name="conv1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(32, 64,stride=1,sparsity = sparsity, name="dw1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(64, 128, stride=2,sparsity = sparsity, name="dw2"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 128,stride = 1,sparsity = sparsity, name="dw3"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 256, stride=2, sparsity = sparsity,name="dw4"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 256,stride = 1,sparsity = sparsity, name="dw5"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 512, stride=2, sparsity = sparsity,name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 512, stride = 1,sparsity = sparsity,name=f"dw7_{i}"))

        self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 1024, stride=2,sparsity = sparsity, name="dw8"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(1024, 1024,stride = 1 ,sparsity = sparsity,name="dw9"))

        self.global_pool = GlobalAveragePooling2D()

        #self.dense_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1024, num_classes)),trainable=True, name="dense_weights" )
        self.dense_weights = SparseTensor([1024,num_classes],sparsity,name="dense")
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")

        self.sparse_tensors = self.get_sparse_tensors()

    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        x = self.global_pool(x)
        #x = tf.matmul(x, self.dense_weights) + self.dense_bias
        x = conv.matmul(x,self.dense_weights) + self.dense_bias
        return tf.nn.softmax(x)

    # TODO: sembra un getter
    def get_sparse_tensors(self):
        prunable = []
        def collect_sparse(module):
            # Include attributes of this module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, SparseTensor):
                    prunable.append(attr)
        collect_sparse(self)
        for submodule in self.submodules:
            collect_sparse(submodule)
        return prunable

    def prune(self, rho):
        prunable = self.sparse_tensors
        tot_pruned = 0
        for i, tensor in enumerate(prunable):
            pruned_i = tensor.prune(rho=rho)
            tot_pruned = tot_pruned + pruned_i
        return tot_pruned

    def update_relative_momentum_contributions(self, tensors):
        total = sum(t.mean_momentum for t in tensors)
        if total == 0:
            equal_share = 1.0 / len(tensors)
            for t in tensors:
                t.relative_momentum_contribution = equal_share
        else:
            for t in tensors:
                t.relative_momentum_contribution = t.mean_momentum / total


    def update_mean_momentums(self, optimizer):
        for sparse_tensor in self.sparse_tensors:
            w = sparse_tensor.values
            idx = optimizer._get_variable_index(w)
            momentum_tensor = optimizer._momentums[idx]
            sparse_tensor.mean_momentum = tf.reduce_mean(tf.abs(momentum_tensor))

    def regrow(self, to_regrow):
        # se vuoi capire perchè esiste regrowth_residual pensa al caso in cui to_regrow = 1

        if to_regrow > self.num_inactive_weights():
            raise ValueError(
                f"Cannot regrow {to_regrow} elements. "
                f"Only {self.num_inactive_weights()} positions available."
            )

        tot_regrown = 0
        remaining_to_allocate  = to_regrow
        active_tensors = self.sparse_tensors

        while remaining_to_allocate  != 0:
            self.update_relative_momentum_contributions(active_tensors)
            total_deficit = 0
            unsaturated_tensors = []
            for t in active_tensors:
                expected_regrow = int(remaining_to_allocate  * t.relative_momentum_contribution)

                actual_regrow = t.regrow(requested_growth=expected_regrow)
                tot_regrown = tot_regrown + actual_regrow
                deficit = expected_regrow - actual_regrow
                total_deficit = total_deficit + deficit
                if not t.is_saturated():
                    unsaturated_tensors.append(t)

            active_tensors = unsaturated_tensors
            remaining_to_allocate  = total_deficit

        residual = to_regrow - tot_regrown
        for t in active_tensors:
            regrown = t.regrow(residual)
            residual = residual-regrown
            if residual == 0:
                return

    def prune_and_regrow(self,rho,optimizer):
        print(f"rho: {rho}")
        self.update_mean_momentums(optimizer)
        self.reset_prune_and_regrow_stats()
        num_pruned = self.prune(rho)
        print(self.prune_summary())
        self.regrow(num_pruned)
        print(self.regrow_summary())

    def reset_prune_and_regrow_stats(self):
        for t in self.sparse_tensors:
            t.reset_prune_and_regrow_stats()

    def num_pruned(self):
        tot_pruned = 0
        for t in self.sparse_tensors:
            tot_pruned = tot_pruned + t.num_pruned
        return tot_pruned

    def num_regrown(self):
        tot_regrown = 0
        for t in self.sparse_tensors:
            tot_regrown = tot_regrown + t.num_regrown
        return tot_regrown

    def prune_summary(self):
        """Returns a detailed summary of pruning statistics for all sparse tensors.

        Returns:
            str: Formatted string containing pruning statistics for each tensor and global summary.
        """

        sep_length = 81

        lines = []
        lines.append("Pruning Summary")
        lines.append("=" * sep_length)

        # Table header
        lines.append(
            f"{'Layer Name':<15} {'Active/Total (b)':>22} {'Pruned':>8} {'Active/Total (a)':>22} {'Sparsity':>10}")
        lines.append("-" * sep_length)

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            pruned = tensor.num_pruned
            active_before_pruning = active + pruned
            sparsity = tensor.tensor_sparsity()

            lines.append(
                f"{tensor.name:<15} {f'{active_before_pruning:,}/{total:,}':>22} {pruned:>8} "
                f"{f'{active:,}/{total:,}':>22} {f'{sparsity:.1%}':>10}"
            )

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Pruned Weights: ", f"{self.num_pruned():,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def regrow_summary(self):
        """Returns a detailed summary of regrowth statistics for all sparse tensors.

        Returns:
            str: Formatted string containing regrowth statistics for each tensor and global summary.
        """
        sep_length = 88
        lines = []
        lines.append("Regrowth Summary")
        lines.append("=" * sep_length)

        # Table Header
        lines.append(
            f"{'Layer Name':<15} {'Momentum':>10} {'Active/Total (b)':>20} "
            f"{'Regrown':>8} {'Active/Total (a)':>20} {'Sparsity':>10}"
        )
        lines.append("-" * sep_length)

        total_regrown = 0

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active_after = tensor.num_active_weights()
            total = tensor.num_weights()
            regrown = tensor.num_regrown
            active_before = active_after - regrown
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"

            lines.append(
                f"{tensor.name:<15} {momentum_pct:>10} {f'{active_before:,}/{total:,}':>20} "
                f"{regrown:>8} {f'{active_after:,}/{total:,}':>20} {f'{sparsity:.1%}':>10}"
            )
            total_regrown += regrown

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Regrown Weights: ", f"{total_regrown:,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def model_sparsity(self):
        return self.num_inactive_weights()/self.num_weights()

    def num_active_weights(self):
        num_active_weights = 0
        for t in self.sparse_tensors:
            num_active_weights = num_active_weights + t.num_active_weights()
        return num_active_weights

    def num_weights(self):
        num_weights = 0
        for t in self.sparse_tensors:
            num_weights = num_weights + t.num_weights()
        return num_weights

    def num_inactive_weights(self):
        return self.num_weights() - self.num_active_weights()
'''

# mobile net sparso base
'''
class MobileNetTF(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity ,name=None):
            super().__init__(name=name)
            self.stride = stride
            #self.conv_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, out_channels)),name="conv")
            self.conv_weights = SparseTensor([3,3,in_channels,out_channels],sparsity, name="conv")
            self.bn = BatchNormalization()

        def __call__(self, x, training=False):
            #x = tf.nn.conv2d(x, self.conv_weights, strides=self.stride, padding="SAME")
            x = conv.sparse_to_dense_conv2d(x,self.conv_weights,self.stride,padding="SAME")
            x = self.bn(x, training=training)
            return tf.nn.relu6(x)
    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity ,name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]

            #self.dw_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(3, 3, in_channels, 1)), trainable=True, name="dw_weights")
            self.dw_weights = SparseTensor([3,3,in_channels,1],sparsity, name="dw_weights")

            #self.pw_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, 1, in_channels, out_channels)),trainable=True, name="pw_weights")
            self.pw_weights = SparseTensor([1,1,in_channels,out_channels],sparsity, name="pw_weights")
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

        def __call__(self, x, training=False):
            #x = tf.nn.depthwise_conv2d(x, self.dw_weights, strides=self.strides, padding="SAME")
            x = conv.sparse_to_dense_depthwise_conv2d(x,self.dw_weights,self.strides,padding="SAME")
            x = self.bn1(x, training=training)
            x = tf.nn.relu6(x)

            #x = tf.nn.conv2d(x, self.pw_weights, strides=1, padding="SAME")
            x = conv.sparse_to_dense_conv2d(x,self.pw_weights,stride=1,padding="SAME")
            x = self.bn2(x, training=training)
            return tf.nn.relu6(x)

    def __init__(self,sparsity, num_classes=8, name=None):
        super().__init__(name=name)
        self.blocks = []

        # Define blocks with explicit channel sizes
        self.blocks.append(MobileNetTF.ConvBlock(3, 32, stride=2, sparsity = sparsity, name="conv1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(32, 64,stride=1,sparsity = sparsity, name="dw1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(64, 128, stride=2,sparsity = sparsity, name="dw2"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 128,stride = 1,sparsity = sparsity, name="dw3"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 256, stride=2, sparsity = sparsity,name="dw4"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 256,stride = 1,sparsity = sparsity, name="dw5"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 512, stride=2, sparsity = sparsity,name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 512, stride = 1,sparsity = sparsity,name=f"dw7_{i}"))

        self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 1024, stride=2,sparsity = sparsity, name="dw8"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(1024, 1024,stride = 1 ,sparsity = sparsity,name="dw9"))

        self.global_pool = GlobalAveragePooling2D()

        #self.dense_weights = tf.Variable(tf.initializers.GlorotUniform()(shape=(1024, num_classes)),trainable=True, name="dense_weights" )
        self.dense_weights = SparseTensor([1024,num_classes],sparsity,name="dense")
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")

    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        x = self.global_pool(x)
        #x = tf.matmul(x, self.dense_weights) + self.dense_bias
        x = conv.matmul(x,self.dense_weights) + self.dense_bias
        return tf.nn.softmax(x)
'''


# mobile net check
'''
class MobileNetTF(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.recompute = recompute_gradient
            self.conv_weights = SparseTensor([3, 3, in_channels, out_channels], sparsity, name="conv")
            self.bn = BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = conv.sparse_to_dense_conv2d(x, self.conv_weights, self.stride, padding="SAME")
                x = self.bn(x, training=training)
                return tf.nn.relu6(x)

            if self.recompute:
                return tf.recompute_grad(forward)(x)
            else:
                return forward(x)

    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]
            self.recompute = recompute_gradient
            self.dw_weights = SparseTensor([3, 3, in_channels, 1], sparsity, name="dw_weights")
            self.pw_weights = SparseTensor([1, 1, in_channels, out_channels], sparsity, name="pw_weights")
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = conv.sparse_to_dense_depthwise_conv2d(x, self.dw_weights, self.strides, padding="SAME")
                x = self.bn1(x, training=training)
                x = tf.nn.relu6(x)

                x = conv.sparse_to_dense_conv2d(x, self.pw_weights, stride=1, padding="SAME")
                x = self.bn2(x, training=training)
                return tf.nn.relu6(x)

            if self.recompute:
                return tf.recompute_grad(forward)(x)
            else:
                return forward(x)

    def __init__(self, sparsity, num_classes=8, recompute_gradient=False, name=None):
        super().__init__(name=name)
        self.recompute = recompute_gradient
        self.blocks = []

        # Define all blocks with recompute flag
        self.blocks.append(MobileNetTF.ConvBlock(3, 32, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="conv1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(32, 64, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(64, 128, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw2"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 128, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw3"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 256, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw4"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 256, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw5"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 512, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 512, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name=f"dw7_{i}"))

        self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 1024, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw8"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(1024, 1024, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw9"))

        self.global_pool = GlobalAveragePooling2D()
        self.dense_weights = SparseTensor([1024, num_classes], sparsity, name="dense")
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")

    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)

        def head(x):
            x = self.global_pool(x)
            x = conv.matmul(x, self.dense_weights) + self.dense_bias
            return tf.nn.softmax(x)

        if self.recompute:
            return tf.recompute_grad(head)(x)
        else:
            return head(x)
'''

# mobile net check, prune and regrow

class MobileNetTF(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.recompute = recompute_gradient
            self.conv_weights = SparseTensor([3, 3, in_channels, out_channels], sparsity, name="conv")
            self.bn = BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = conv.sparse_to_dense_conv2d(x, self.conv_weights, self.stride, padding="SAME")
                x = self.bn(x, training=training)
                return tf.nn.relu6(x)

            if self.recompute:
                return tf.recompute_grad(forward)(x)
            else:
                return forward(x)

    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]
            self.recompute = recompute_gradient
            self.dw_weights = SparseTensor([3, 3, in_channels, 1], sparsity, name="dw_weights")
            self.pw_weights = SparseTensor([1, 1, in_channels, out_channels], sparsity, name="pw_weights")
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = conv.sparse_to_dense_depthwise_conv2d(x, self.dw_weights, self.strides, padding="SAME")
                x = self.bn1(x, training=training)
                x = tf.nn.relu6(x)

                x = conv.sparse_to_dense_conv2d(x, self.pw_weights, stride=1, padding="SAME")
                x = self.bn2(x, training=training)
                return tf.nn.relu6(x)

            if self.recompute:
                return tf.recompute_grad(forward)(x)
            else:
                return forward(x)

    def __init__(self, sparsity, num_classes=8, recompute_gradient=False, name=None):
        super().__init__(name=name)
        self.recompute = recompute_gradient
        self.blocks = []

        # Define all blocks with recompute flag
        self.blocks.append(MobileNetTF.ConvBlock(3, 32, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="conv1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(32, 64, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw1"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(64, 128, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw2"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 128, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw3"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(128, 256, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw4"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 256, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw5"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(256, 512, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 512, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name=f"dw7_{i}"))

        self.blocks.append(MobileNetTF.DepthwiseConvBlock(512, 1024, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw8"))
        self.blocks.append(MobileNetTF.DepthwiseConvBlock(1024, 1024, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw9"))

        self.global_pool = GlobalAveragePooling2D()
        self.dense_weights = SparseTensor([1024, num_classes], sparsity, name="dense")
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")

        self.sparse_tensors = self.get_sparse_tensors()

    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)

        def head(x):
            x = self.global_pool(x)
            x = conv.matmul(x, self.dense_weights) + self.dense_bias
            return tf.nn.softmax(x)

        if self.recompute:
            return tf.recompute_grad(head)(x)
        else:
            return head(x)

    def __str__(self):
        lines = []
        lines.append("ResNet50_sparse2 Model Summary")
        lines.append("=" * 80)

        # Table Header
        lines.append(f"{'Layer Name':<30} {'Momentum Contribution':>25} {'Sparsity':>10} {'Active / Total':>15}")
        lines.append("-" * 80)

        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"
            sparsity_str = f"{sparsity:.1%}"

            lines.append(
                f"{tensor.name:<30} {momentum_pct:>25} {sparsity_str:>10} {f'{active:,} / {total:,}':>15}"
            )

        lines.append("-" * 80)

        # Model-level summary
        model_sparsity = self.model_sparsity()
        model_active = self.num_active_weights()
        model_total = self.num_weights()

        lines.append(f"{'Overall Model Sparsity:':<30} {model_sparsity:.1%}")
        lines.append(f"{'Total Active / Total Weights:':<30} {model_active:,} / {model_total:,}")

        lines.append("=" * 80)
        return "\n".join(lines)

    # TODO: sembra un getter
    def get_sparse_tensors(self):
        prunable = []
        def collect_sparse(module):
            # Include attributes of this module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, SparseTensor):
                    prunable.append(attr)
        collect_sparse(self)
        for submodule in self.submodules:
            collect_sparse(submodule)
        return prunable

    def prune(self, rho):
        prunable = self.sparse_tensors
        tot_pruned = 0
        for i, tensor in enumerate(prunable):
            pruned_i = tensor.prune(rho=rho)
            tot_pruned = tot_pruned + pruned_i
        return tot_pruned

    def update_relative_momentum_contributions(self, tensors):
        total = sum(t.mean_momentum for t in tensors)
        if total == 0:
            equal_share = 1.0 / len(tensors)
            for t in tensors:
                t.relative_momentum_contribution = equal_share
        else:
            for t in tensors:
                t.relative_momentum_contribution = t.mean_momentum / total

    def update_mean_momentums(self, optimizer):
        for sparse_tensor in self.sparse_tensors:
            w = sparse_tensor.values
            idx = optimizer._get_variable_index(w)
            momentum_tensor = optimizer._momentums[idx]
            sparse_tensor.mean_momentum = tf.reduce_mean(tf.abs(momentum_tensor))

    def regrow(self, to_regrow):

        # se vuoi capire perchè esiste regrowth_residual pensa al caso in cui to_regrow = 1
        if to_regrow > self.num_inactive_weights():
            raise ValueError(
                f"Cannot regrow {to_regrow} elements. "
                f"Only {self.num_inactive_weights()} positions available."
            )

        tot_regrown = 0
        remaining_to_allocate  = to_regrow
        active_tensors = self.sparse_tensors

        while remaining_to_allocate  != 0:
            self.update_relative_momentum_contributions(active_tensors)
            total_deficit = 0
            unsaturated_tensors = []
            for t in active_tensors:
                expected_regrow = int(remaining_to_allocate  * t.relative_momentum_contribution)

                actual_regrow = t.regrow(requested_growth=expected_regrow)
                tot_regrown = tot_regrown + actual_regrow
                deficit = expected_regrow - actual_regrow
                total_deficit = total_deficit + deficit
                if not t.is_saturated():
                    unsaturated_tensors.append(t)

            active_tensors = unsaturated_tensors
            remaining_to_allocate  = total_deficit

        residual = to_regrow - tot_regrown
        for t in active_tensors:
            regrown = t.regrow(residual)
            residual = residual-regrown
            if residual == 0:
                return

    def prune_and_regrow(self,rho,optimizer):
        print(f"rho: {rho}")
        self.update_mean_momentums(optimizer)
        self.reset_prune_and_regrow_stats()
        num_pruned = self.prune(rho)
        print(self.prune_summary())
        self.regrow(num_pruned)
        print(self.regrow_summary())

    def reset_prune_and_regrow_stats(self):
        for t in self.sparse_tensors:
            t.reset_prune_and_regrow_stats()

    def num_pruned(self):
        tot_pruned = 0
        for t in self.sparse_tensors:
            tot_pruned = tot_pruned + t.num_pruned
        return tot_pruned

    def num_regrown(self):
        tot_regrown = 0
        for t in self.sparse_tensors:
            tot_regrown = tot_regrown + t.num_regrown
        return tot_regrown

    def prune_summary(self):
        """Returns a detailed summary of pruning statistics for all sparse tensors.

        Returns:
            str: Formatted string containing pruning statistics for each tensor and global summary.
        """

        sep_length = 81

        lines = []
        lines.append("Pruning Summary")
        lines.append("=" * sep_length)

        # Table header
        lines.append(
            f"{'Layer Name':<15} {'Active/Total (b)':>22} {'Pruned':>8} {'Active/Total (a)':>22} {'Sparsity':>10}")
        lines.append("-" * sep_length)

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active = tensor.num_active_weights()
            total = tensor.num_weights()
            pruned = tensor.num_pruned
            active_before_pruning = active + pruned
            sparsity = tensor.tensor_sparsity()

            lines.append(
                f"{tensor.name:<15} {f'{active_before_pruning:,}/{total:,}':>22} {pruned:>8} "
                f"{f'{active:,}/{total:,}':>22} {f'{sparsity:.1%}':>10}"
            )

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Pruned Weights: ", f"{self.num_pruned():,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def regrow_summary(self):
        """Returns a detailed summary of regrowth statistics for all sparse tensors.

        Returns:
            str: Formatted string containing regrowth statistics for each tensor and global summary.
        """
        sep_length = 88
        lines = []
        lines.append("Regrowth Summary")
        lines.append("=" * sep_length)

        # Table Header
        lines.append(
            f"{'Layer Name':<15} {'Momentum':>10} {'Active/Total (b)':>20} "
            f"{'Regrown':>8} {'Active/Total (a)':>20} {'Sparsity':>10}"
        )
        lines.append("-" * sep_length)

        total_regrown = 0

        # Update momentum contributions
        self.update_relative_momentum_contributions(self.sparse_tensors)

        for tensor in self.sparse_tensors:
            active_after = tensor.num_active_weights()
            total = tensor.num_weights()
            regrown = tensor.num_regrown
            active_before = active_after - regrown
            sparsity = tensor.tensor_sparsity()
            momentum_pct = f"{tensor.relative_momentum_contribution:.1%}"

            lines.append(
                f"{tensor.name:<15} {momentum_pct:>10} {f'{active_before:,}/{total:,}':>20} "
                f"{regrown:>8} {f'{active_after:,}/{total:,}':>20} {f'{sparsity:.1%}':>10}"
            )
            total_regrown += regrown

        lines.append("-" * sep_length)

        # Right-aligned global summary
        def right_align(label, value):
            left = f"{label}"
            right = f"{value}"
            total_width = sep_length
            return f"{left}{' ' * (total_width - len(left) - len(right))}{right}"

        lines.append(right_align("Total Regrown Weights: ", f"{total_regrown:,}"))
        lines.append(right_align("Model Sparsity: ", f"{self.model_sparsity():.1%}"))
        lines.append(
            right_align("Total Active/Total Weights: ", f"{self.num_active_weights():,}/{self.num_weights():,}"))

        lines.append("=" * sep_length)
        return "\n".join(lines)

    def model_sparsity(self):
        return self.num_inactive_weights()/self.num_weights()

    def num_active_weights(self):
        num_active_weights = 0
        for t in self.sparse_tensors:
            num_active_weights = num_active_weights + t.num_active_weights()
        return num_active_weights

    def num_weights(self):
        num_weights = 0
        for t in self.sparse_tensors:
            num_weights = num_weights + t.num_weights()
        return num_weights

    def num_inactive_weights(self):
        return self.num_weights() - self.num_active_weights()










def main():
    X_train, y_train = funzioni.load_bloodmnist_subset(); X_val = X_train; y_val = y_train
    #(X_train, y_train), (X_test, y_test), (X_val, y_val) = funzioni.load_bloodmnist_224()

    #model = ResNet50_sparse2(sparsity= 0.8, recompute = True)
    ##model = ResNet50_2(num_classes=8)

    model = MobileNetTF(sparsity=0.8, recompute_gradient=True)

    max_iter = 130000
    funzioni.train(model,
                   X_train,
                   y_train,
                   X_val,
                   y_val,
                   epochs=100,
                   max_iter = max_iter,
                   batch_size=32,
                   lr=0.001,
                   live_plotting=False,
                   weights_chekpoint_stride=200,
                   prune_and_regrow_stride=209,
                   rho0 = 0.5,
                   microbatch_size = 16)

if __name__ == '__main__':
    main()



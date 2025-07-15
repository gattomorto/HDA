import tensorflow as tf
import numpy as np
import keras
import random
import os
import math
import cv2
import matplotlib.pyplot as plt
import gc
import time
from tensorflow.python.ops.gen_sparse_ops import sparse_reorder, sparse_tensor_dense_mat_mul, sparse_to_dense
import numpy as np
import cv2
from skimage.util import view_as_windows

#se togli questo i risultati dei test potrebbero cambiare
SEED = 0
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class SparseModel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        #self.sparse_tensors = []
        #self.sparse_tensors = self._collect_sparse_tensors()

    def __str__(self):
        lines = []
        lines.append("Model Sparsity Summary")
        lines.append("=" * 80)

        # Table Header
        lines.append(f"{'Layer Name':<30} {'Momentum Contribution':>25} {'Sparsity':>10} {'Active / Total':>15}")
        lines.append("-" * 80)

        self.update_relative_momentum_contributions(self.sparse_tensors())

        for tensor in self.sparse_tensors():
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


    def sparse_tensors(self):
        return self._collect_sparse_tensors()

    '''@sparse_tensors.setter
    def sparse_tensors(self, value):
        self._sparse_tensors = value'''

    def _collect_sparse_tensors(self):
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
        prunable = self.sparse_tensors()
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
        for sparse_tensor in self.sparse_tensors():
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
        active_tensors = self.sparse_tensors()

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
        #print(f"rho: {rho}")
        self.update_mean_momentums(optimizer)
        self.reset_prune_and_regrow_stats()
        num_pruned = self.prune(rho)
        #print(self.prune_summary())
        self.regrow(num_pruned)
        #print(self.regrow_summary())

    def reset_prune_and_regrow_stats(self):
        for t in self.sparse_tensors():
            t.reset_prune_and_regrow_stats()

    def num_pruned(self):
        tot_pruned = 0
        for t in self.sparse_tensors():
            tot_pruned = tot_pruned + t.num_pruned
        return tot_pruned

    def num_regrown(self):
        tot_regrown = 0
        for t in self.sparse_tensors():
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
        self.update_relative_momentum_contributions(self.sparse_tensors())

        for tensor in self.sparse_tensors():
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
        self.update_relative_momentum_contributions(self.sparse_tensors())

        for tensor in self.sparse_tensors():
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
        for t in self.sparse_tensors():
            num_active_weights = num_active_weights + t.num_active_weights()
        return num_active_weights

    def num_weights(self):
        num_weights = 0
        for t in self.sparse_tensors():
            num_weights = num_weights + t.num_weights()
        return num_weights

    def num_inactive_weights(self):
        return self.num_weights() - self.num_active_weights()

    def is_rho_large_enough(self,rho):
        print("phi:", int(self.num_active_weights() * rho))
        return int(self.num_active_weights() * rho) > len(self.sparse_tensors())

class SparseTensor(tf.Module):
    def __init__(self, *args,  name=None):
        def random_indices(shape, sparsity):
            total = np.prod(shape)
            nnz = int((1.0 - sparsity) * total)  # Number of non-zero elements
            rng = np.random.default_rng(0)
            chosen = rng.choice(total, size=nnz, replace=False)
            unraveled = np.stack(np.unravel_index(chosen, shape), axis=-1)  # shape: (nnz, len(shape))
            flat_sorted_order = np.ravel_multi_index(unraveled.T, shape)
            sorted_indices = unraveled[np.argsort(flat_sorted_order)]
            return tf.constant(sorted_indices, dtype=tf.int64), nnz

        super().__init__(name=name)
        self.relative_momentum_contribution = 0
        self.mean_momentum = 0
        self.num_pruned = 0
        self.num_regrown = 0
        self.num_active_weights_before_pruning=0
        if len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], (float,int)):
            shape, sparsity = args
            self.sparsity = sparsity
            self.shape = tf.convert_to_tensor(shape,dtype=tf.int64)
            self.indices, nnz = random_indices(shape, sparsity)
            #
            if len(shape) == 4:
                '''fan_in = shape[0] * shape[1] * shape[2]
                stddev = np.sqrt(2.0 / fan_in)
                initializer = keras.initializers.TruncatedNormal(stddev=stddev)'''

                fan_in = shape[0] * shape[1] * shape[2]
                fan_out = shape[3]  # output channels
                limit = np.sqrt(6.0 / (fan_in + fan_out))  # Xavier/Glorot formula
                initializer = keras.initializers.RandomUniform(minval=-limit, maxval=limit)
            elif len(shape) == 2:

                fan_in = shape[0]
                stddev = np.sqrt(2.0 / fan_in)
                initializer = keras.initializers.TruncatedNormal(stddev=stddev)

                '''fan_in = shape[0]
                fan_out = shape[1]
                limit = np.sqrt(6 / (fan_in + fan_out))
                initializer = keras.initializers.RandomUniform(minval=-limit, maxval=limit)'''


            else:
                initializer = keras.initializers.RandomNormal()
                #exit()

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
            raise TypeError("Invalid arguments for SparseTensor initialization.")

    def __eq__(self, other):
        return tf.reduce_all(tf.equal(self.to_tf_dense(), other.to_tf_dense()))

    def __str__(self):
        dense = self.to_tf_dense()
        return str(dense.numpy())

    def __repr__(self):
        return str(self)

    def rowmajor_reorder(self):
        '''tf_sparse = self.to_tf_sparse()
        tf_sparse_reordered =  tf.sparse.reorder(tf_sparse)
        self.indices = tf_sparse_reordered.indices
        self.values.assign(tf_sparse_reordered.values)'''
        sp_tensor = sparse_reorder(self.indices, self.values, self.shape)
        self.indices = sp_tensor.output_indices
        self.values.assign(sp_tensor.output_values)

    def to_tf_sparse(self):
        return tf.sparse.SparseTensor(self.indices, self.values, self.shape)

    def to_tf_dense(self):
        #from tensorflow.python.ops.gen_sparse_ops import sparse_to_dense
        #return tf.sparse.to_dense(self.to_tf_sparse())
        return sparse_to_dense(self.indices,self.shape,self.values,default_value=tf.constant(0.0),validate_indices=False)

    def prune(self, rho):
        '''
        :param rho: fraction of nonzero weights to prune
        :return: number of pruned weights
        #ATTENZIONE: gli indici potrebbero essere disordinati
        '''

        self.num_active_weights_before_pruning = self.num_active_weights()

        # sort by absolute value
        idx_sorted = tf.argsort(tf.math.abs(self.values))
        values_sorted = tf.gather(self.values, idx_sorted)
        indices_sorted = tf.gather(self.indices, idx_sorted)

        split_idx = tf.cast(tf.math.ceil(tf.cast(tf.shape(values_sorted)[0], tf.float32) * rho), tf.int32)

        if split_idx==0:
            return 0

        new_indices = indices_sorted[split_idx:]
        new_values = values_sorted[split_idx:]
        self.indices = new_indices
        self.values = tf.Variable(new_values, trainable=True)

        num_pruned = split_idx.numpy()
        self.num_pruned = self.num_pruned + num_pruned
        return num_pruned

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
            initializer = keras.initializers.HeNormal()
            new_values = initializer(shape=[actual_regrow], dtype=self.values.dtype)
            #new_values = tf.zeros([actual_regrow], dtype=self.values.dtype)

            # Update indices and values
            self.indices = tf.concat([self.indices, new_indices], axis=0)

            self.values = tf.Variable(
                tf.concat([self.values, new_values], axis=0),
                trainable=True
            )

            # Convert to numpy only once at the end
            actual_regrow_np = actual_regrow.numpy()
            self.num_regrown = self.num_regrown + actual_regrow_np
        else:
            actual_regrow_np = 0

        self.rowmajor_reorder()
        return actual_regrow_np

    def reset_prune_and_regrow_stats(self):
        self.num_regrown = 0
        self.num_pruned = 0

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

class ResNet(SparseModel):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, recompute=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            self.recompute = recompute #recompute grad

            self.w1 = SparseTensor([1, 1, in_channels, filters], sparsity, name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = keras.layers.BatchNormalization(name="bn1")

            self.w2 = SparseTensor([3, 3, filters, filters], sparsity, name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 =keras.layers.BatchNormalization(name="bn2")

            self.w3 = SparseTensor([1, 1, filters, 4 * filters], sparsity, name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = SparseTensor([1, 1, in_channels, 4 * filters], sparsity, name="w_sc_M")
                self.b_sc = tf.Variable(tf.zeros([4 * filters]), name="b_sc")
                self.bn_sc = keras.layers.BatchNormalization(name="bn_sc")

        def __call__(self, x, training=False):
            def forward(x):
                shortcut = x
                if self.conv_shortcut:
                    shortcut = sparse_to_dense_conv2d(x, self.w_sc, stride=self.stride, padding="SAME") + self.b_sc
                    shortcut = self.bn_sc(shortcut, training=training)

                x = sparse_to_dense_conv2d(x, self.w1, stride=self.stride, padding="SAME") + self.b1
                x = self.bn1(x, training=training)
                x = tf.nn.relu(x)

                x = sparse_to_dense_conv2d(x, self.w2, stride=1, padding="SAME") + self.b2
                x = self.bn2(x, training=training)
                x = tf.nn.relu(x)

                x = sparse_to_dense_conv2d(x, self.w3, stride=1, padding="SAME") + self.b3
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
                ResNet.ConvBlock(sparsity, in_channels, filters, stride=stride1, conv_shortcut=True, recompute=recompute, name="block1")
            )
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet.ConvBlock(sparsity, 4 * filters, filters, stride=1, conv_shortcut=False, recompute=recompute, name=f"block{i}")
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
        self.bn1 = keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, recompute=recompute, name="conv2")
        self.stage3 = ResNet.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, recompute=recompute, name="conv3")
        self.stage4 = ResNet.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, recompute=recompute, name="conv4")
        self.stage5 = ResNet.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, recompute=recompute, name="conv5")

        self.fc_w = SparseTensor([2048, num_classes], sparsity, name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

        #self.sparse_tensors = self._collect_sparse_tensors()

    def __call__(self, x, training=False):
        def stem(x):
            x = sparse_to_dense_conv2d(x, self.conv1_w, stride=2, padding="SAME") + self.conv1_b
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = self.pool(x)
            return x

        def head(x):
            x = tf.reduce_mean(x, axis=[1, 2])  # Global Average Pooling
            logits = sparse_to_dense_matmul(x, self.fc_w) + self.fc_b
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

class MobileNet224(SparseModel):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.recompute = recompute_gradient
            self.conv_weights = SparseTensor([3, 3, in_channels, out_channels], sparsity, name="conv")
            self.bn = keras.layers.BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = sparse_to_dense_conv2d(x, self.conv_weights, self.stride, padding="SAME")
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
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = sparse_to_dense_depthwise_conv2d(x, self.dw_weights, self.strides, padding="SAME")
                x = self.bn1(x, training=training)
                x = tf.nn.relu6(x)

                x = sparse_to_dense_conv2d(x, self.pw_weights, stride=1, padding="SAME")
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
        self.blocks.append(MobileNet224.ConvBlock(3, 32, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="conv1"))
        self.blocks.append(MobileNet224.DepthwiseConvBlock(32, 64, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw1"))
        self.blocks.append(MobileNet224.DepthwiseConvBlock(64, 128, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw2"))
        self.blocks.append(MobileNet224.DepthwiseConvBlock(128, 128, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw3"))
        self.blocks.append(MobileNet224.DepthwiseConvBlock(128, 256, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw4"))
        self.blocks.append(MobileNet224.DepthwiseConvBlock(256, 256, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw5"))
        self.blocks.append(MobileNet224.DepthwiseConvBlock(256, 512, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw6"))

        for i in range(5):
            self.blocks.append(MobileNet224.DepthwiseConvBlock(512, 512, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name=f"dw7_{i}"))

        self.blocks.append(MobileNet224.DepthwiseConvBlock(512, 1024, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw8"))
        self.blocks.append(MobileNet224.DepthwiseConvBlock(1024, 1024, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw9"))

        self.global_pool = keras.layers.GlobalAveragePooling2D()
        self.dense_weights = SparseTensor([1024, num_classes], sparsity, name="dense")
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")


    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)

        def head(x):
            x = self.global_pool(x)
            x = sparse_to_dense_matmul(x, self.dense_weights) + self.dense_bias
            return tf.nn.softmax(x)

        if self.recompute:
            return tf.recompute_grad(head)(x)
        else:
            return head(x)

class MobileNet32(SparseModel):
    class ConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.recompute = recompute_gradient
            self.conv_weights = SparseTensor([3, 3, in_channels, out_channels], sparsity, name="conv")
            self.bn = keras.layers.BatchNormalization()
        def __call__(self, x, training=False):
            def forward(x):
                x = sparse_to_dense_conv2d(x, self.conv_weights, self.stride, padding="SAME")
                x = self.bn(x, training=training)
                return tf.nn.relu6(x)
            return tf.recompute_grad(forward)(x) if self.recompute else forward(x)

    class DepthwiseConvBlock(tf.Module):
        def __init__(self, in_channels, out_channels, stride, sparsity, recompute_gradient=False, name=None):
            super().__init__(name=name)
            self.strides = [1, stride, stride, 1]
            self.recompute = recompute_gradient
            self.dw_weights = SparseTensor([3, 3, in_channels, 1], sparsity, name="dw_weights")
            self.pw_weights = SparseTensor([1, 1, in_channels, out_channels], sparsity, name="pw_weights")
            self.bn1 = keras.layers.BatchNormalization()
            self.bn2 = keras.layers.BatchNormalization()

        def __call__(self, x, training=False):
            def forward(x):
                x = sparse_to_dense_depthwise_conv2d(x, self.dw_weights, self.strides, padding="SAME")
                x = self.bn1(x, training=training)
                x = tf.nn.relu6(x)
                x = sparse_to_dense_conv2d(x, self.pw_weights, stride=1, padding="SAME")
                x = self.bn2(x, training=training)
                return tf.nn.relu6(x)
            return tf.recompute_grad(forward)(x) if self.recompute else forward(x)

    def __init__(self, sparsity, num_classes=8, recompute_gradient=False, name=None):
        super().__init__(name=name)
        self.recompute = recompute_gradient
        self.blocks = []

        # Blocks for 32x32 input resolution
        self.blocks.append(MobileNet32.ConvBlock(3, 32, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="conv1"))
        self.blocks.append(MobileNet32.DepthwiseConvBlock(32, 64, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw1"))
        self.blocks.append(MobileNet32.DepthwiseConvBlock(64, 128, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw2"))  # 32→16
        self.blocks.append(MobileNet32.DepthwiseConvBlock(128, 128, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw3"))
        self.blocks.append(MobileNet32.DepthwiseConvBlock(128, 256, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw4"))  # 16→8
        self.blocks.append(MobileNet32.DepthwiseConvBlock(256, 256, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw5"))
        self.blocks.append(MobileNet32.DepthwiseConvBlock(256, 512, stride=2, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw6"))  # 8→4

        self.blocks.append(MobileNet32.DepthwiseConvBlock(512, 1024, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw8"))
        self.blocks.append(MobileNet32.DepthwiseConvBlock(1024, 1024, stride=1, sparsity=sparsity, recompute_gradient=recompute_gradient, name="dw9"))

        self.global_pool = keras.layers.GlobalAveragePooling2D()
        self.dense_weights = SparseTensor([1024, num_classes], sparsity, name="dense")
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")
    def __call__(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)

        def head(x):
            x = self.global_pool(x)
            x = sparse_to_dense_matmul(x, self.dense_weights) + self.dense_bias
            return tf.nn.softmax(x)

        return tf.recompute_grad(head)(x) if self.recompute else head(x)




#############################################################################################################

def load_bloodmnist_224(patching):
    try:
        data = np.load("bloodmnist_224.npz")
    except:
        data = np.load("/content/drive/MyDrive/hda/bloodmnist_224.npz")

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

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    y_val = keras.utils.to_categorical(y_val)

    if patching:
        # 16->x96
        # 32->x24
        X_train, y_train, discarded_patches, discarded_labels = extract_violet_patches(X_train, y_train, patch_size=32, stride=32)
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        '''for i in range(len(discarded_patches)):
            show_image(discarded_patches,discarded_labels,index = i)
        exit()'''

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

def load_bloodmnist_subset(patching):
    data = np.load("bloodmnist_subset.npz")
    #data = np.load(f"{folder_path}/bloodmnist_224.npz")

    X = data['X']
    y = data['y']
    print(f"Loaded subset: {X.shape}, {y.shape}")
    Xc = X.copy()
    yc = y.copy()

    if patching:
        X, y, discarded_patches, discarded_labels = extract_violet_patches(X, y, patch_size=32, stride=32)
        indices = np.random.permutation(len(X))
        X_train = X[indices]
        y_train = y[indices]

    return (X, y), (Xc, yc), (Xc, yc)

# test standard
def test(model, X, y, batch_size=2000):
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

def test2(model, X, y, patch_size=32, stride=16, batch_size=32):
    N = X.shape[0]
    num_classes = y.shape[1]
    summed_probs_per_image = np.zeros((N, num_classes))
    has_violet = np.zeros(N, dtype=bool)  # track if at least one patch was found per image

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X_batch = X[start:end]
        y_batch = y[start:end]

        batch_retained_patches = []
        patch_image_indices = []

        # Extract patches per image in batch
        for i in range(X_batch.shape[0]):
            retained_patches, _, _, _ = extract_violet_patches(
                X_batch[i:i+1], y_batch[i:i+1], patch_size, stride
            )
            if len(retained_patches) > 0:
                batch_retained_patches.append(retained_patches)
                patch_image_indices.extend([start + i] * len(retained_patches))
                has_violet[start + i] = True

        if batch_retained_patches:
            batch_retained_patches = np.concatenate(batch_retained_patches, axis=0)
            preds = model(batch_retained_patches, training=False).numpy()
            for idx, img_idx in enumerate(patch_image_indices):
                summed_probs_per_image[img_idx] += preds[idx]

    if not np.any(has_violet):
        raise Exception("No violet patches found in the dataset.")

    # Predict class for images that had violet patches
    predicted_classes = np.full(N, -1)
    predicted_classes[has_violet] = np.argmax(summed_probs_per_image[has_violet], axis=1)
    true_classes = np.argmax(y, axis=1)

    # Compute accuracy only on images that had violet patches
    accuracy = np.mean(predicted_classes[has_violet] == true_classes[has_violet])
    return accuracy


def sparse_to_dense_conv2d(input, sp_filter, stride=1, padding='SAME'):
    #dense_filter = tf.sparse.to_dense(sp_filter.to_tf_sparse())
    dense_filter = sp_filter.to_tf_dense()
    return tf.nn.conv2d(input,dense_filter,stride,padding)

def sparse_to_dense_depthwise_conv2d(input, sp_filter, strides, padding='SAME'):
    #dense_filter = tf.sparse.to_dense(sp_filter.to_tf_sparse())
    dense_filter = sp_filter.to_tf_dense()
    return tf.nn.depthwise_conv2d(input,dense_filter,strides,padding)

def sparse_to_dense_matmul(X,Y_sp):
    Y_dense = Y_sp.to_tf_dense()
    return tf.matmul(X, Y_dense, b_is_sparse=False)

def plot_overlapped_curves_old(file_list, start=0):

    plt.figure(figsize=(10, 6))

    for file_path in file_list:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()[start:]  # Skip the first 'start' lines
                data = [float(line.strip()) for line in lines if line.strip()]
                plt.plot(data, label=file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Overlapped Curves from Text Files')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_overlapped_curves(file_list, start, flag):
    """
    Plots curves from text files containing rows of: time, loss, acc.

    Args:
        file_list (list): List of file paths.
        start (int or float):
            - If flag == 'loss', skip the first 'start' lines.
            - If flag == 'acc', filter out data where time < start.
        flag (str): Either 'loss' or 'acc'.
    """
    plt.figure(figsize=(10, 6))

    for file_path in file_list:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

                time_vals = []
                loss_vals = []
                acc_vals = []

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        continue
                    t, l, a = map(float, parts)
                    time_vals.append(t)
                    loss_vals.append(l)
                    acc_vals.append(a)

                if flag == 'loss':
                    loss_vals = loss_vals[start:]
                    plt.plot(loss_vals, label=file_path,alpha=0.5)
                elif flag == 'acc':
                    # Filter based on time
                    filtered = [(t, a) for t, a in zip(time_vals, acc_vals) if t >= start]
                    if filtered:
                        t_filtered, a_filtered = zip(*filtered)
                        plt.plot(t_filtered, a_filtered, label=file_path,alpha=0.5)
                else:
                    print(f"Unknown flag '{flag}', skipping file {file_path}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    plt.xlabel('Index' if flag == 'loss' else 'Time')
    plt.ylabel('Loss' if flag == 'loss' else 'Accuracy')
    plt.title('Overlapped Curves: ' + ('Loss' if flag == 'loss' else 'Accuracy'))
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_column_from_files(file_paths, start=0.0):
    """
    Plots column 1 vs column 0 from multiple text files starting from a given x-value.

    Each file is assumed to have at least 2 numeric columns.
    Column 0 is used for the x-axis, column 1 for the y-axis.

    Parameters:
        file_paths (list of str): Paths to the input text files.
        start (float): Minimum value of column 0 (x-axis) from which to start plotting.

    Raises:
        ValueError: If the file does not contain at least 2 columns.
    """
    for path in file_paths:
        try:
            data = np.loadtxt(path)
            if data.shape[1] < 2:
                raise ValueError(f"File '{path}' does not contain at least 2 columns.")

            # Filter rows based on column 0 values
            mask = data[:, 0] >= start
            if not np.any(mask):
                raise ValueError(f"No data in file '{path}' with column 0 >= {start}.")

            x = data[mask, 0]
            y = data[mask, 1]
            label = os.path.basename(path)
            plt.plot(x, y, label=label)
            #plt.scatter(x, y, label=label, s=10)

        except Exception as e:
            print(f"Error processing '{path}': {e}")

    plt.xlabel("Column 0 (x-axis)")
    plt.ylabel("Column 1 (y-axis)")
    plt.title(f"Plot of Column 1 vs Column 0 (starting from x ≥ {start})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_image(X, y, index=0):
    image = X[index]
    plt.imshow(image)
    plt.title(y[index])
    plt.axis('off')
    plt.show()


def extract_violet_patches(X, y, patch_size, stride):
    N, H, W, C = X.shape
    num_patches_h = (H - patch_size) // stride + 1
    num_patches_w = (W - patch_size) // stride + 1

    # Initialize lists for patches
    retained_patches = []
    retained_labels = []
    discarded_patches = []
    discarded_labels = []

    # Define violet color range in HSV
    lower_violet = np.array([130, 50, 50])  # HSV range for violet
    upper_violet = np.array([160, 255, 255])

    for n in range(N):
        # Convert normalized RGB (0-1) to RGB (0-255) for OpenCV
        img_rgb_255 = (X[n] * 255).astype(np.uint8)

        # Convert image to HSV
        img_hsv = cv2.cvtColor(img_rgb_255, cv2.COLOR_RGB2HSV)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Extract patch coordinates
                h_start = i * stride
                h_end = h_start + patch_size
                w_start = j * stride
                w_end = w_start + patch_size

                # Get the patch in RGB (keep original normalized values)
                patch_rgb = X[n, h_start:h_end, w_start:w_end, :]

                # Get the patch in HSV (from scaled version)
                patch_hsv = img_hsv[h_start:h_end, w_start:w_end, :]

                # Create mask for violet pixels
                violet_mask = cv2.inRange(patch_hsv, lower_violet, upper_violet)

                # Check if ANY pixel is violet (at least one pixel)
                has_violet = np.any(violet_mask > 0)

                if has_violet:
                    retained_patches.append(patch_rgb)
                    retained_labels.append(y[n])
                else:
                    discarded_patches.append(patch_rgb)
                    discarded_labels.append(y[n])

    # Convert lists to numpy arrays
    retained_patches = np.array(retained_patches) if retained_patches else np.empty((0, patch_size, patch_size, 3))
    retained_labels = np.array(retained_labels) if retained_labels else np.empty((0, y.shape[1]))
    discarded_patches = np.array(discarded_patches) if discarded_patches else np.empty((0, patch_size, patch_size, 3))
    discarded_labels = np.array(discarded_labels) if discarded_labels else np.empty((0, y.shape[1]))

    return retained_patches, retained_labels, discarded_patches, discarded_labels

def train(model, X_tr, y_tr, X_val, y_val,X_test, y_test, max_epochs, max_iter, max_time, batch_size, lr,
          prune_and_regrow_frequency, test_frequency, patience, rho0, microbatch_size):

    def data_generator(X, y, batch_size):
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]

    #log_file1 = open("training_log1.txt", 'a')
    log_file2 = open("training_log2.txt", 'a')
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)

    training_start_time = time.time()
    accumulated_elapsed_time = 0.0
    last_test_training_time = 0.0

    it = 0
    best_avg_loss = float('inf')
    patience_counter = 0
    total_stride_loss = 0
    stride_loss_count = 0
    num_tests = 0
    best_acc_val = 0

    for epoch in range(max_epochs):
        #print(f"\nEpoch {epoch + 1}")

        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_generator(
                lambda: data_generator(X_tr, y_tr, batch_size),
                output_signature=(
                    tf.TensorSpec(shape=(None,) + X_tr.shape[1:], dtype=X_tr.dtype),
                    tf.TensorSpec(shape=(None,) + y_tr.shape[1:], dtype=y_tr.dtype)
                )
            )
        for x_batch, y_batch in dataset:
            if microbatch_size is None:
                with tf.GradientTape() as tape:
                    preds = model(x_batch, training=True)
                    micro_loss = loss_fn(y_batch, preds)
                grads = tape.gradient(micro_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                loss_val = micro_loss.numpy()
            else:
                '''microbatches = tf.data.Dataset.from_tensor_slices((x_batch, y_batch)).batch(microbatch_size)
                accum_grads = [tf.zeros_like(var) for var in model.trainable_variables]
                total_micro_loss = 0.0
                num_microbatches = 0

                for x_micro, y_micro in microbatches:'''
                accum_grads = [tf.zeros_like(var) for var in model.trainable_variables]
                total_micro_loss = 0.0
                batch_size_current = tf.shape(x_batch)[0]
                num_microbatches = tf.math.ceil(batch_size_current / microbatch_size).numpy()

                for i in range(0, batch_size_current, microbatch_size):
                    end_idx = tf.minimum(i + microbatch_size, batch_size_current)
                    x_micro = x_batch[i:end_idx]
                    y_micro = y_batch[i:end_idx]

                    with tf.GradientTape() as tape:
                        preds = model(x_micro, training=True)
                        micro_loss = loss_fn(y_micro, preds)
                    grads = tape.gradient(micro_loss, model.trainable_variables)
                    accum_grads = [acc_g + g for acc_g, g in zip(accum_grads, grads)]
                    total_micro_loss += micro_loss.numpy()
                    #num_microbatches += 1

                avg_grads = [g / num_microbatches for g in accum_grads]
                optimizer.apply_gradients(zip(avg_grads, model.trainable_variables))
                loss_val = total_micro_loss / num_microbatches

            it += 1
            total_stride_loss += loss_val
            stride_loss_count += 1
            '''if tf.config.list_physical_devices('GPU'):
                print(f"Peak Memory: {tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1024 ** 2:.1f} MB")
            else:
                print(f"Peak Memory: {tf.config.experimental.get_memory_info('CPU:0')['peak'] / 1024 ** 2:.1f} MB")'''

            # Recalculate training-only time
            current_elapsed = accumulated_elapsed_time + (time.time() - training_start_time)
            '''if it % 1 == 0:
                #print(f"E: {epoch}, Num Tests: {num_tests}, lr: {optimizer.learning_rate.numpy()},  Best Loss: {best_avg_loss}, Best Acc: {best_acc_val}, Step {it}, Loss: {loss_val}, Elapsed: {current_elapsed}")
                log_file1.write(f"{current_elapsed} {loss_val} {best_acc_val}\n")
                log_file1.flush()'''

            if it % prune_and_regrow_frequency == 0:
                rho = rho0 ** (int(it / prune_and_regrow_frequency))
                if model.is_rho_large_enough(rho):
                    print("Prune & Regrow")
                    model.prune_and_regrow(rho, optimizer)
                    optimizer = keras.optimizers.Adam(learning_rate=float(optimizer.learning_rate.numpy()))
                    best_avg_loss = float('inf')
                    patience_counter = 0
                else:
                    print("Prune & Regrow Aborted")

            current_elapsed = accumulated_elapsed_time + (time.time() - training_start_time)
            if current_elapsed - last_test_training_time >= test_frequency:
                # Stop training clock
                test_start_time = time.time()
                accumulated_elapsed_time += (test_start_time - training_start_time)

                #acc_val = test2(model, X_val, y_val, batch_size=32)
                acc_val = test2(model, X_test, y_test, batch_size=32)
                #acc_val = -1
                test_duration = time.time() - test_start_time

                # Update test time and resume training clock
                last_test_training_time += test_frequency
                training_start_time = time.time()

                num_tests += 1

                avg_stride_loss = total_stride_loss / stride_loss_count
                total_stride_loss = 0
                stride_loss_count = 0

                #print(f"Avg Stride Loss: {avg_stride_loss}")

                if avg_stride_loss < best_avg_loss:
                    best_avg_loss = avg_stride_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        new_lr = optimizer.learning_rate.numpy() * 0.5
                        optimizer.learning_rate.assign(new_lr)
                        print(f"Reducing LR to {new_lr:.6f}")
                        patience_counter = 0

                if acc_val > best_acc_val:
                    best_acc_val = acc_val

                print(f"Patience counter: {patience_counter}")
                print(f"Time Step: {current_elapsed}, Best Accuracy: {best_acc_val}, Accuracy: {acc_val}, it: {it}, Loss: {loss_val}, Best Avg Loss: {best_avg_loss}, Avg Loss: {avg_stride_loss} lr: {optimizer.learning_rate.numpy()}")
                log_file2.write(f"{current_elapsed} {best_acc_val} {best_avg_loss} {avg_stride_loss}\n")
                log_file2.flush()

            if it == max_iter:
                print("max iter reached")
                #log_file.close()
                return

            if current_elapsed > max_time:
                print("max time reached")
                #log_file.close()
                return

    #log_file.close()

#TODO: resnet: recompute_gradient
def main():

    #file_list = ['./runs/r43b.txt','./runs/r50b.txt','./runs/r51b.txt','./runs/r52b.txt','./runs/r55b.txt','./runs/r45b.txt']
    file_list = ['./runs/r45a.txt','./runs/r78a.txt']
    #file_list = ['./runs/r67a.txt','./runs/r66a.txt']
    plot_overlapped_curves(file_list, start=0, flag= "loss")
    #plot_column_from_files(file_list,0)
    exit()

    if tf.config.list_physical_devices('GPU'):
      (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_bloodmnist_224(patching=True)
      test_frequency = 60*2.5
      batch_size = 32
      prune_and_regrow_frequency = 50
      microbatch_size = 16

    else:
        (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_bloodmnist_subset(patching=True)
        test_frequency = 60*2.5
        batch_size = 32
        prune_and_regrow_frequency = 50
        microbatch_size = 16


    # CAMBIA TEST, CAMBIA INIZIALIZZAZIONE
    #model = MobileNet224(sparsity=0.8,recompute_gradient=True)
    model = MobileNet32(sparsity=0.8, recompute_gradient=True)

    train(model, X_train, y_train, X_val, y_val, X_test, y_test, max_epochs=100000, max_iter=10000000, max_time=60* 60* 2, batch_size=batch_size,
          lr=0.001, prune_and_regrow_frequency=prune_and_regrow_frequency, test_frequency=test_frequency, patience=3, rho0=0.5,
          microbatch_size=microbatch_size)

if __name__ == '__main__':
    main()



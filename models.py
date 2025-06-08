import tensorflow as tf
import numpy as np
import keras
import random
import os
from tensorflow.python.ops.gen_sparse_ops import sparse_reorder, sparse_tensor_dense_mat_mul, sparse_to_dense

folder_path = '/content/drive/MyDrive/hda'
import sys
sys.path.append(folder_path)

SEED = 0
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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
        #TODO: indices and shapes tf variables?
        if len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], (float,int)):
            shape, sparsity = args
            self.sparsity = sparsity
            self.shape = tf.convert_to_tensor(shape,dtype=tf.int64)
            self.indices, nnz = random_indices(shape, sparsity)
            initializer = keras.initializers.HeNormal()
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
        #TODO: default_value=tf.constant(0.0) per half precision?
        #TODO: validation_indices = False
        return sparse_to_dense(self.indices,self.shape,self.values,default_value=tf.constant(0.0),validate_indices=True)

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

# checkpointing, con prune & regrow
class ResNet50_sparse2(tf.Module):
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
        self.bn1 = keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, recompute=recompute, name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, recompute=recompute, name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, recompute=recompute, name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, recompute=recompute, name="conv5")

        self.fc_w = SparseTensor([2048, num_classes], sparsity, name="fc_w_M")
        self.fc_b = tf.Variable(tf.zeros([num_classes]), name="fc_b")

        self.sparse_tensors = self._collect_sparse_tensors()

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

# mobile net, check, prune and regrow
class MobileNetTF(tf.Module):
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

        self.global_pool = keras.layers.GlobalAveragePooling2D()
        self.dense_weights = SparseTensor([1024, num_classes], sparsity, name="dense")
        self.dense_bias = tf.Variable(tf.zeros([num_classes]), trainable=True, name="dense_bias")

        self.sparse_tensors = self._collect_sparse_tensors()

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


def load_bloodmnist_224():
    data = np.load("bloodmnist_224.npz")
    #data = np.load("/content/drive/MyDrive/hda/bloodmnist_224.npz")

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

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

def load_bloodmnist_subset():
    data = np.load("bloodmnist_subset.npz")
    #data = np.load(f"{folder_path}/bloodmnist_224.npz")

    X = data['X']
    y = data['y']
    print(f"Loaded subset: {X.shape}, {y.shape}")
    return X, y

def test(model, X, y, batch_size=32):
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


def sparse_to_dense_conv2d(input, sp_filter, stride=1, padding='SAME'):
    #dense_filter = tf.sparse.to_dense(sp_filter.to_tf_sparse())
    dense_filter = sp_filter.to_tf_dense()
    return tf.nn.conv2d(input,dense_filter,stride,padding)

def sparse_to_dense_depthwise_conv2d(input, sp_filter, strides, padding='SAME'):
    #dense_filter = tf.sparse.to_dense(sp_filter.to_tf_sparse())
    dense_filter = sp_filter.to_tf_dense()
    return tf.nn.depthwise_conv2d(input,dense_filter,strides,padding)

def sparse_to_dense_matmul(X,Y_sp):
    '''
    Y_dense = sparse_to_dense(Y_sp.indices, Y_sp.shape,Y_sp.values,
                        default_value=0,
                        validate_indices=True)
    '''
    Y_dense = Y_sp.to_tf_dense()
    #TODO: b_is_sparse = True dà risultati diversi -- forse gli indici non sono corretti
    return tf.matmul(X, Y_dense, b_is_sparse=False)


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
    test_stride,
    patience,
    rho0,
    microbatch_size
):
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(batch_size)

    it = 0

    best_loss = float('inf')
    patience_counter = 0
    step_losses = []
    step_numbers = []



    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        epoch_loss = 0
        num_batches = 0

        for step, (x_batch, y_batch) in enumerate(dataset):
            if microbatch_size is None:
                with tf.GradientTape() as tape:
                    #TODO: logits non va bene
                    logits = model(x_batch, training=True)
                    loss = loss_fn(y_batch, logits)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                loss_val = loss.numpy()

            else:
                microbatches = tf.data.Dataset.from_tensor_slices((x_batch, y_batch)).batch(microbatch_size)
                accum_grads = [tf.zeros_like(var) for var in model.trainable_variables]
                total_loss = 0.0
                num_microbatches = 0

                for x_micro, y_micro in microbatches:
                    with tf.GradientTape() as tape:
                        logits = model(x_micro, training=True)
                        loss = loss_fn(y_micro, logits)
                    grads = tape.gradient(loss, model.trainable_variables)
                    accum_grads = [acc_g + g for acc_g, g in zip(accum_grads, grads)]
                    total_loss += loss.numpy()
                    num_microbatches += 1

                avg_grads = [g / num_microbatches for g in accum_grads]
                optimizer.apply_gradients(zip(avg_grads, model.trainable_variables))
                loss_val = total_loss / num_microbatches

            epoch_loss += loss_val
            step_losses.append(loss_val)
            step_numbers.append(it)
            num_batches += 1

            it += 1


            print(f"E: {epoch}, BL: {best_loss}, Step {it}, Loss: {loss_val}")
            with open("training_log.txt", 'a') as f:
                f.write(f"{loss_val}\n")

            #print(f"Peak Memory: {tf.config.experimental.get_memory_info('CPU:0')['peak'] / 1024 ** 2:.1f} MB")

            if it % prune_and_regrow_stride == 0:
                rho = rho0 ** (int(it / prune_and_regrow_stride))
                print("phi:",int(model.num_active_weights()*rho))
                if int(model.num_active_weights()*rho) > len(model.sparse_tensors):
                    print("Prune & Regrow")
                    model.prune_and_regrow(rho, optimizer)
                    optimizer = keras.optimizers.Adam(learning_rate=float(optimizer.learning_rate.numpy()))
                    best_loss = float('inf')
                    patience_counter=0
                else:
                    print("Prune & Regrow Aborted")

            if it % test_stride == 0:
                #acc_tr = test(model, X_tr, y_tr)
                acc_tr = -1
                acc_val = test(model, X_val, y_val)
                print(f"Step {it}, Accuracy Train: {acc_tr:.3f},  Accuracy Val: {acc_val:.3f}")


            if it == max_iter:
                print("max iter reached")
                return

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} Avg Loss: {avg_epoch_loss}")

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


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


def plot_overlapped_curves(file_list, start=0):
    """
    Plots overlapped curves from a list of .txt files, skipping the first 'start' lines of each file.

    Parameters:
        file_list (list of str): List of paths to .txt files.
        start (int): Number of initial lines to skip in each file.
    """
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


def main():
    file_list = ['./runs/r6.txt','./runs/r10.txt']
    plot_overlapped_curves(file_list,start=2300)

    exit()

    X_train, y_train = load_bloodmnist_subset(); X_val = X_train; y_val = y_train
    #(X_train, y_train), (X_test, y_test), (X_val, y_val) = load_bloodmnist_224()

    model = ResNet50_sparse2(sparsity= 0.8, recompute = False)

    #model = MobileNetTF(sparsity=0.8, recompute_gradient=False)

    max_iter = 130000
    train(model,
           X_train,
           y_train,
           X_val,
           y_val,
           epochs = 100,
           max_iter = max_iter,
           batch_size = 1,
           lr = 0.001,
           patience = 3,
           prune_and_regrow_stride = 1,
           test_stride = 10000,
           rho0 = 0.5,
           microbatch_size = None
           )

if __name__ == '__main__':
    main()



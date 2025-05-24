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
import gc
#tf.keras.backend.set_floatx('float16')
SEED = 0
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

'''
# versione seria sparsa checkpoint
class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut

            self.w1 = funzioni.SparseTensor([1, 1, in_channels, filters], sparsity, name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")

            self.w2 = funzioni.SparseTensor([3, 3, filters, filters], sparsity, name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")

            self.w3 = funzioni.SparseTensor([1, 1, filters, 4 * filters], sparsity, name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = funzioni.SparseTensor([1, 1, in_channels, 4 * filters], sparsity, name="w_sc_M")
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

        self.conv1_w = funzioni.SparseTensor([7, 7, 3, 64], sparsity, name="conv1_w_M")
        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, name="conv5")

        self.fc_w = funzioni.SparseTensor([2048, num_classes], sparsity, name="fc_w_M")
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


#versione seria sparsa senza check
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
            self.w1 = funzioni.SparseTensor([1, 1, in_channels, filters],sparsity,name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
            #-----------------------------------------------------------------
            #self.w2 = tf.Variable(initializer([3, 3, filters, filters]), name="w2")
            self.w2 = funzioni.SparseTensor([3, 3, filters, filters],sparsity,name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
            #--------------------------------------------------------------------
            #self.w3 = tf.Variable(initializer([1, 1, filters, 4 * filters]), name="w3")
            self.w3 = funzioni.SparseTensor([1, 1, filters, 4 * filters],sparsity,name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                #self.w_sc = tf.Variable(initializer([1, 1, in_channels, 4 * filters]), name="w_sc")
                self.w_sc = funzioni.SparseTensor([1, 1, in_channels, 4 * filters],sparsity, name="w_sc_M")
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


    def __init__(self, sparsity, num_classes=1000, name=None):
        super().__init__(name=name)
        self.num_classes = num_classes

        #initializer = tf.keras.initializers.HeNormal()

        # Initial conv
        #self.conv1_w = tf.Variable(initializer([7, 7, 3, 64]), name="conv1_w")
        self.conv1_w = funzioni.SparseTensor([7, 7, 3, 64],sparsity,name = "conv1_w_M")
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
        self.fc_w = funzioni.SparseTensor([2048, num_classes],sparsity ,name="fc_w_M")
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
#ResNet50_sparse & ResNet50_2(non sparse) ResNet50_sparse2(checkpointed) devono dare gli stessi risultati -- servono solo per il debug
class ResNet50_sparse(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut
            initializer = tf.keras.initializers.HeNormal()

            # Conv layers
            #self.w1 = tf.Variable(initializer([1, 1, in_channels, filters]), name="w1")
            self.w1 = funzioni.SparseTensor(v4.create_tensor_row_major(1, in_channels, filters),name="w1mio")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
            #-----------------------------------------------------------------
            #self.w2 = tf.Variable(initializer([3, 3, filters, filters]), name="w2")
            self.w2 = funzioni.SparseTensor(v4.create_tensor_row_major(3, filters, filters),name="w2mio")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
            #--------------------------------------------------------------------
            #self.w3 = tf.Variable(initializer([1, 1, filters, 4 * filters]), name="w3")
            self.w3 = funzioni.SparseTensor(v4.create_tensor_row_major(1, filters, 4*filters),name="w3mio")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                #self.w_sc = tf.Variable(initializer([1, 1, in_channels, 4 * filters]), name="w_sc")
                self.w_sc = funzioni.SparseTensor(v4.create_tensor_row_major(1, in_channels, 4 * filters), name="w_scmio")
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
        self.conv1_w = funzioni.SparseTensor(v4.create_tensor_row_major(7, 3, 64), name="conv1_wmio")
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
        self.fc_w = funzioni.SparseTensor([2048, num_classes],0 ,name="fc_wmio")
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
        self.fc_w =  tf.Variable(funzioni.SparseTensor([2048, num_classes],0 ,name="fc_wmio").to_tf_dense(), name="fc_wmio")
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
class ResNet50_sparse2(tf.Module):
    class ConvBlock(tf.Module):
        def __init__(self, sparsity, in_channels, filters, stride=1, conv_shortcut=True, name=None):
            super().__init__(name=name)
            self.stride = stride
            self.conv_shortcut = conv_shortcut

            self.w1 = funzioni.SparseTensor(v4.create_tensor_row_major(1, in_channels, filters),name="w1mio")
            #self.w1 = funzioni.SparseTensor([1, 1, in_channels, filters], sparsity, name="w1_M")
            self.b1 = tf.Variable(tf.zeros([filters]), name="b1")
            self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")

            self.w2 = funzioni.SparseTensor(v4.create_tensor_row_major(3, filters, filters),name="w2mio")
            #self.w2 = funzioni.SparseTensor([3, 3, filters, filters], sparsity, name="w2_M")
            self.b2 = tf.Variable(tf.zeros([filters]), name="b2")
            self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")

            self.w3 = funzioni.SparseTensor(v4.create_tensor_row_major(1, filters, 4*filters),name="w3mio")
            #self.w3 = funzioni.SparseTensor([1, 1, filters, 4 * filters], sparsity, name="w3_M")
            self.b3 = tf.Variable(tf.zeros([4 * filters]), name="b3")
            self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")

            if conv_shortcut:
                self.w_sc = funzioni.SparseTensor(v4.create_tensor_row_major(1, in_channels, 4 * filters), name="w_scmio")

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
        self.conv1_w = funzioni.SparseTensor(v4.create_tensor_row_major(7, 3, 64), name="conv1_wmio")

        self.conv1_b = tf.Variable(tf.zeros([64]), name="conv1_b")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1_bn")

        self.pool = lambda x: tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")

        self.stage2 = ResNet50_sparse2.ResNetStack(64, 64, 3, stride1=1, sparsity=sparsity, name="conv2")
        self.stage3 = ResNet50_sparse2.ResNetStack(256, 128, 4, stride1=2, sparsity=sparsity, name="conv3")
        self.stage4 = ResNet50_sparse2.ResNetStack(512, 256, 6, stride1=2, sparsity=sparsity, name="conv4")
        self.stage5 = ResNet50_sparse2.ResNetStack(1024, 512, 3, stride1=2, sparsity=sparsity, name="conv5")

        self.fc_w = funzioni.SparseTensor([2048, num_classes],0 ,name="fc_wmio")
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
'''


#TODO: x = tf.nn.conv2d(x, self.w1, strides=..., padding=...) + self.b1 vs x = tf.nn.bias_add(tf.nn.conv2d(x, self.w1, strides=..., padding=...), self.b1)
#TODO: dropput
#TODO: SparseCategoricalCrossentropy(from_logits=True) cos'è?
#TODO: You're recreating BatchNormalization layers directly in the block constructors — that's fine, but remember: They must be reused correctly during training and inference. You're doing this right — just keep this in mind when saving/loading
#ResNet50_original ResNet50_keras sono versioni base non checkpointed e non sparse
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


# ConvBlockWithCheckpoint & ResNet50_keras_check sono insieme
class ConvBlockWithCheckpoint(tf.keras.layers.Layer):
    def __init__(self, filters, stride, shortcut_conv):
        super().__init__()
        self.filters = filters
        self.stride = stride
        self.shortcut_conv = shortcut_conv

        # Predefine the layers
        if self.shortcut_conv:
            self.shortcut_layer = layers.Conv2D(4 * filters, 1, strides=stride)
            self.shortcut_bn = layers.BatchNormalization()
        else:
            self.shortcut_layer = None

        self.conv1 = layers.Conv2D(filters, 1, strides=stride)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(4 * filters, 1)
        self.bn3 = layers.BatchNormalization()

        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        @tf.recompute_grad
        def _forward(x):
            shortcut = x
            if self.shortcut_conv:
                shortcut = self.shortcut_layer(shortcut)
                shortcut = self.shortcut_bn(shortcut, training=training)

            x = self.conv1(x)
            x = self.bn1(x, training=training)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x, training=training)
            x = self.relu(x)

            x = self.conv3(x)
            x = self.bn3(x, training=training)

            x = self.add([x, shortcut])
            x = self.relu(x)
            return x

        return _forward(inputs)
def ResNet50_keras_check(input_shape=(224, 224, 3), num_classes=1000):
    def resnet_stack(x, filters, blocks, stride1):
        x = ConvBlockWithCheckpoint(filters, stride1, shortcut_conv=True)(x)
        for _ in range(1, blocks):
            x = ConvBlockWithCheckpoint(filters, stride=1, shortcut_conv=False)(x)
        return x

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = resnet_stack(x, 64, 3, stride1=1)
    x = resnet_stack(x, 128, 4, stride1=2)
    x = resnet_stack(x, 256, 6, stride1=2)
    x = resnet_stack(x, 512, 3, stride1=2)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
'''

def main():
    X_train, y_train = funzioni.load_bloodmnist_subset(); X_val = X_train; y_val = y_train

    #(X_train, y_train), (X_test, y_test), (X_val, y_val) = funzioni.load_bloodmnist_224()
    #model = ResNet50_sparse(sparsity= 0, num_classes=8)
    #model = ResNet50_2( num_classes=8)
    model = ResNet50_original(num_classes=8)
    #model = ResNet50_sparse2_check(sparsity= 0.99, num_classes=8)
    #model = ResNet50_sparse2(sparsity= 0, num_classes=8)
    #model = ResNet50_keras(num_classes=8)
    #model = ResNet50_keras_check(num_classes=8)

    '''sp = tf.io.parse_tensor(tf.io.read_file('sp.bytes'), out_type=tf.float32)
    de = tf.io.parse_tensor(tf.io.read_file('de.bytes'), out_type=tf.float32)
    print(tf.reduce_max(tf.abs(sp - de)))
    exit()'''


    '''t = model.trainable_variables
    trainable_count = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print("Total number of trainable scalars:", trainable_count)
    #for var in model.trainable_variables:
    #print(var.name)'''

    #23497864 sp = 0
    #23550984 keras
    #  261250
    #   26568 sp = 1

    funzioni.train(model, X_train, y_train, X_val,y_val, epochs=100, batch_size=1, lr=0.001, live_plotting=False,prune_and_regrow_step=3000)

if __name__ == '__main__':
    main()


#ciao
import funzioni
import tensorflow as tf
import random
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.losses import CategoricalCrossentropy

SEED = 0
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class ResNet50_sparse(tf.Module):
    sparsity = 0.8
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

            self.blocks.append(ResNet50.ConvBlock(in_channels, filters, stride=stride1, conv_shortcut=True, name="block1"))
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50.ConvBlock(4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
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
        self.stage2 = ResNet50.ResNetStack(64, 64, 3, stride1=1, name="conv2")
        self.stage3 = ResNet50.ResNetStack(256, 128, 4, stride1=2, name="conv3")
        self.stage4 = ResNet50.ResNetStack(512, 256, 6, stride1=2, name="conv4")
        self.stage5 = ResNet50.ResNetStack(1024, 512, 3, stride1=2, name="conv5")

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


#TODO: x = tf.nn.conv2d(x, self.w1, strides=..., padding=...) + self.b1 vs x = tf.nn.bias_add(tf.nn.conv2d(x, self.w1, strides=..., padding=...), self.b1)
#TODO: dropput
#TODO: SparseCategoricalCrossentropy(from_logits=True) cos'è?
#TODO: You're recreating BatchNormalization layers directly in the block constructors — that's fine, but remember: They must be reused correctly during training and inference. You're doing this right — just keep this in mind when saving/loading
class ResNet50(tf.Module):
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

            self.blocks.append(ResNet50.ConvBlock(in_channels, filters, stride=stride1, conv_shortcut=True, name="block1"))
            for i in range(2, blocks + 1):
                self.blocks.append(
                    ResNet50.ConvBlock(4 * filters, filters, stride=1, conv_shortcut=False, name=f"block{i}")
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
        self.stage2 = ResNet50.ResNetStack(64, 64, 3, stride1=1, name="conv2")
        self.stage3 = ResNet50.ResNetStack(256, 128, 4, stride1=2, name="conv3")
        self.stage4 = ResNet50.ResNetStack(512, 256, 6, stride1=2, name="conv4")
        self.stage5 = ResNet50.ResNetStack(1024, 512, 3, stride1=2, name="conv5")

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



def main():

    X_train, y_train = funzioni.load_bloodmnist_subset()
    model = ResNet50_sparse( num_classes=8)
    funzioni.train(model, X_train, y_train, epochs=10, batch_size=32, lr=0.001, prune_and_regrow_step=3000)


if __name__ == '__main__':
    main()



import os
import sys
from guppy import hpy

import numpy as np
import matplotlib.pyplot as plt
import psutil
import tensorflow as tf
from cnn_utils import *
from memory_profiler import profile
from memory_profiler import memory_usage

# Define a function to wrap with tf.recompute_grad
@tf.recompute_grad
def my_function1(x):
    print("Running f1")
    y = tf.math.sin(x)
    y = tf.math.exp(y)
    y = tf.math.log(y + 1.0)
    return y # vettore

def my_function2(x): #vettore
    print("Running f2")
    y = tf.math.cos(x)
    y = tf.math.square(y)
    return y #vettore

def my_function3(x): #vettore
    print("Running f3")
    return tf.reduce_sum(y) #scalare


x = tf.Variable([0.1, 0.2, 0.3], dtype=tf.float32)

# durante il forward pas tape ricorda tutti i risultati intermedi per tutte le
# funzioni che non hanno @tf.recompute_grad. Quando chiami tape.gradient ricalcola
# i risultati intermedi solo per funzioni che.

print("frw")
with tf.GradientTape() as tape:
    y = my_function1(x)
    z = my_function2(y)
    t = my_function3(z)
print("bkw")
grads = tape.gradient(y, x)

print("Gradient:", grads.numpy())




#!/usr/bin/env python
# coding: utf-8

# TensorFlow and tf.keras
import tensorflow as tf
import json
import sys
from tensorflow import keras

# Helper libraries
import numpy as np

test_dimsize = 100

print("TF version :%s" % tf.__version__)

_input  = np.arange(100, 100+test_dimsize)
_output = _input * 100 + 5

model = keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1]),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer=keras.optimizers.Adam(0.01),
        loss='mse')

model.fit(_input, _output, epochs=2000, batch_size=10, verbose=1)
model.summary()

print("prediction test")
test_input = [1,2,3,4,5]
print("input : ", test_input)
test_output = model.predict(test_input)
print("output: ", test_output.tolist())

#1. save model as tensorflow saved_model
model.save('saved_model/my_model')

#2. save model as Keras H5 model
model.save('saved_model/my_model.h5')

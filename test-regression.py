#!/usr/bin/env python
# coding: utf-8

# TensorFlow and tf.keras
import tensorflow as tf
import json
from tensorflow import keras

# Helper libraries
import numpy as np

test_dimsize = 100

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

print(tf.__version__)

_input  = np.arange(test_dimsize)
_output = _input*10+5

print(_input)
print(_output)

_input = _input + np.random.randn(test_dimsize)
_input = trunc(_input, decs=3)

_output = _output + np.random.randn(test_dimsize)
_output = trunc(_output, decs=3)

print(_input)
print(_output)

model = keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1]),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer='adam',
        loss='mse')

model.summary()

model.fit(_input, _output, epochs=1000, batch_size=10, verbose=0)
weight = model.get_weights()

print("prediction test")
test_input = [-2,-1,2,3,4,5]
print(test_input)
print(model.predict(test_input))

print("weight len : %d" % len(weight))
for i in range(len(weight)):
    print("weight[%d]" % i)
    print(weight[i])
    print("")

model.save('saved_model/my_model')

with open("student_file.json", "w") as json_file:

    json.dump(model.to_json(), json_file)

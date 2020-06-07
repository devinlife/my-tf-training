#!/usr/bin/env python
# coding: utf-8

# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np

# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="./saved_model/my_model.quant.tflite")
interpreter = tf.lite.Interpreter(model_path="./saved_model/my_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = np.array(np.random.randint(0,1000, size=input_shape), dtype=np.float32)

#input_data = np.array([[1]], dtype=np.float32)
print("input : %s" % input_data)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("output : %s" % output_data)

#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import h5py

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tflite-mymodel.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get details for each layer
all_layers_details = interpreter.get_tensor_details()

f = h5py.File("tflite-mymodel.hdf5", "w")

for layer in all_layers_details:
     # to create a group in an hdf5 file
     grp = f.create_group(str(layer['index']))

     # to store layer's metadata in group's metadata
     grp.attrs["name"] = layer['name']
     grp.attrs["shape"] = layer['shape']
     # grp.attrs["dtype"] = all_layers_details[i]['dtype']
     grp.attrs["quantization"] = layer['quantization']

     # to store the weights in a dataset
     grp.create_dataset("weights", data=interpreter.get_tensor(layer['index']))

f.close()

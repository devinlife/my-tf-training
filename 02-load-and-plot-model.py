#!/usr/bin/env python
# coding: utf-8

# TensorFlow and tf.keras
import tensorflow as tf
import json
import sys
import pprint
from tensorflow import keras

# Helper libraries
import numpy as np

loaded = tf.keras.models.load_model('saved_model/my_model')

weight = loaded.get_weights()

#1. print model weights
print("weight len : %d" % len(weight))
for i in range(len(weight)):
    print("weight[%d]" % i)
    print(weight[i])
    print("")

#2. save a model as png
tf.keras.utils.plot_model(
    loaded, to_file='saved_model/my_model.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
)

#3. save a model as json
json_string = loaded.to_json()
with open('saved_model/my_model.json', 'w') as json_file:
    pprint.pprint(json.loads(json_string), json_file)
sys.exit()

#4. run netron with model
#not working with TF2.0 pb file

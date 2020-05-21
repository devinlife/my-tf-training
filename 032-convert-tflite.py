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

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
tflite_model = converter.convert()
open("saved_model/my_model.tflite", "wb").write(tflite_model)

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
open("saved_model/my_model.quant.tflite", "wb").write(tflite_quant_model)

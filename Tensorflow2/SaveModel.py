# 使用 SavedModel 格式
import os
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()
physical_devices = tf.config.list_physical_devices('GPU')

for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)

# file = tf.keras.utils.get_file(
#    "grace_hopper.jpg",
#    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")

file = './Datasets/grace_hopper.jpg'

img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
plt.imshow(img)
plt.axis('off')
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...])

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
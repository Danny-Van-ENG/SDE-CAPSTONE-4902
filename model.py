"""
Model.py

Created by Zachary Smith and Danny Van
Based on Resnet model variant: https://www.kaggle.com/code/utkarshsaxenadn/asl-alphabet-recognition-resnet50v2-acc-97

"""

import cv2  # Import OpenCV for image processing
import sys  # Import for time
import os  # Import for reading files
import tensorflow as tf  # Import tensorflow for Inception Net's backend
from keras.preprocessing.image import ImageDataGenerator as ImgDataGen
from keras.applications import ResNet50V2

from keras.models import Sequential, load_model
from keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout
from tensorflow.python.client import device_lib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root_path = 'C:/GitHub/SDE-CAPSTONE-4902/data/asl_alphabet_train/asl_alphabet_train'
test_path = 'C:/GitHub/SDE-CAPSTONE-4902/data/asl_alphabet_test/asl_alphabet_test'
class_names = sorted(os.listdir(root_path))
n_classes = len(class_names)
print(f"Class Names : \n{class_names}\n")

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory

gen = ImgDataGen(rescale=1. / 255, rotation_range=10, validation_split=0.2)
train_ds = gen.flow_from_directory(root_path, target_size=(256, 256), batch_size=32, subset='training',
                                   class_mode='binary')
valid_ds = gen.flow_from_directory(root_path, target_size=(256, 256), batch_size=32, subset='validation',
                                   class_mode='binary')


name = 'ResNet50V2'
base = ResNet50V2(include_top=False, input_shape=(256, 256, 3))
base.trainable = False

model = Sequential([
    base,
    GAP(),  # Global Average Pooling 2D
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax'),
], name=name)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
# K.tensorflow_backend._get_available_gpus()
model.fit(train_ds,
          validation_data=valid_ds,
          epochs=15)
model.save('C:/GitHub/SDE-CAPSTONE-4902/models')
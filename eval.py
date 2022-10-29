import numpy as np

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator as ImgDataGen

model_path = 'C:/GitHub/SDE-CAPSTONE-4902/models'
test_path = 'C:/GitHub/SDE-CAPSTONE-4902/data/asl_alphabet_test/asl_alphabet_test'

model = keras.models.load_model(model_path)
test_datagen = ImgDataGen(rescale=1. / 255, rotation_range=10, validation_split=0.2)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
)

predictions = model.predict_classes(X_test)
print("Predictions done...")
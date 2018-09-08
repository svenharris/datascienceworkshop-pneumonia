from keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_IMAGES, TEST_IMAGES

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_IMAGES,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    directory=TEST_IMAGES,
    target_size=(150, 150),
    batch_size=32,
)



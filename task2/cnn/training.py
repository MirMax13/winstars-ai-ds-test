from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classifier import AnimalCNNClassifier
IMG_SIZE = 128

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)
                              

train_ds = train_gen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
)

test_ds= test_gen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
)

model = AnimalCNNClassifier(img_size=IMG_SIZE, num_classes=train_ds.num_classes)
model.train(train_ds, test_ds, epochs=10)
model.save("task2/image_classification/model/animal_cnn_model")

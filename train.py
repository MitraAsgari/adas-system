import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import extract_boxes, load_data

# Data generator
class DataGenerator(Sequence):
    def __init__(self, image_files, data_dir, img_size=(128, 128), batch_size=32, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.image_files = image_files
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.image_files))
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.image_files[k] for k in batch_indexes]
        X, y = self.__data_generation(batch_files)
        if self.augment:
            X, y = next(self.datagen.flow(X, y, batch_size=self.batch_size))
        return X, y

    def __data_generation(self, batch_files):
        X = np.empty((self.batch_size, *self.img_size, 3))
        y = np.empty((self.batch_size, 4), dtype=float)

        for i, img_file in enumerate(batch_files):
            img_path = os.path.join(self.data_dir, img_file)
            if img_file.endswith('.jpg'):  # Image file
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    X[i,] = img / 255.0  # Normalize image
            elif img_file.endswith('.xml'):  # XML label file
                xml_path = os.path.join(self.data_dir, img_file)
                boxes = extract_boxes(xml_path)
                if boxes:
                    y[i] = boxes[0]  # Assuming only one bounding box, modify if more than one
                else:
                    y[i] = [0, 0, 0, 0]  # If no bounding box found

        return X, y

def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = '/content/data/export'
    train_files, test_files = load_data(data_dir)
    
    train_generator = DataGenerator(train_files, data_dir, batch_size=32, augment=True)
    test_generator = DataGenerator(test_files, data_dir, batch_size=32)
    
    input_shape = (128, 128, 3)
    model = create_model(input_shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=[early_stopping])

    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy}')

    # Save model
    model.save('object_detection_model.h5')

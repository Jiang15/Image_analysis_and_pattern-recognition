import keras
import skimage
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K 
import json
import os
import numpy as np
from utils import replicate

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x, y, extend_max=6, batch_size=64, shuffle=True):
        'Initialization'
        self.batch_size = batch_size

        self.image_gen = ImageDataGenerator(
            rotation_range=360,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=[0.8, 1.8]
        ).flow(x, y=y, batch_size=batch_size, shuffle=shuffle)

        self.extend_max = extend_max

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.image_gen)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Get data
        X, y = self.image_gen[index]

        # Add some random padding
#         X = replicate(
#             X,
#             left=np.random.randint(self.extend_max),
#             right=np.random.randint(self.extend_max),
#             up=np.random.randint(self.extend_max),
#             down=np.random.randint(self.extend_max)
#         )
        
    
        return X, y

    def on_epoch_end(self):
        self.image_gen.on_epoch_end()
        
        
        
############ MODEL ###############
class CNNModel:
    def __init__(self, model_dir='.', num_classes=14, load=False, verbose=True):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.json_model_file=model_dir + '/json_model.json'
        self.model_weights_file=model_dir + '/model_weights.h5'

        if load:
            self.load()
        else:
            self.model = Sequential([
                # Conv layers
                Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(None, None, 1)),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.20),

                Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.20),

                Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.20),

                # Flatten(),
                Lambda(lambda x: K.mean(x, axis=[1, 2])),

                # Densely connected layers
                Dense(128, activation='relu'),
                BatchNormalization(),

                # Output layer
                Dense(num_classes, activation='softmax')
            ])

            # compile with adam optimizer & categorical_crossentropy loss function
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
        if verbose:
            print(self.model.summary())
    
    def fit(self, x, y, epochs=50, batch_size=64):
        # get train and validation splits
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.10, shuffle=True)

        # instantiate the data generator
        train_generator = DataGenerator(x_train, y=y_train, batch_size=batch_size)

        # train
        self.history = self.model.fit(train_generator, epochs=epochs, validation_data=(x_valid, y_valid))
        
    def save(self):
        # save the model architecture
        with open(self.json_model_file, "w") as json_file:
            json.dump(self.model.to_json(), json_file)

        # save the weights
        self.model.save_weights(self.model_weights_file)
        
    def load(self):
        # load the model architecture
        with open(self.json_model_file,'r') as f:
            json_model = json.load(f)

        # load the weights
        self.model = model_from_json(json_model)
        self.model.load_weights(self.model_weights_file)
        
        # compile with adam optimizer & categorical_crossentropy loss function
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
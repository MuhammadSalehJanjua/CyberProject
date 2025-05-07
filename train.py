import os
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create models directory if not exists
os.makedirs('models', exist_ok=True)

# Build fresh model (no weights loaded)
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/kaggle_dataset',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/kaggle_dataset',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Compile and train
model.compile(optimizer='adam', 
            loss='binary_crossentropy',
            metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save weights
model.save_weights('models/deepfake_xception.weights.h5')
print("Training complete! Weights saved to models/deepfake_xception.h5")
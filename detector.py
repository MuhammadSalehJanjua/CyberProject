import cv2
import numpy as np
from tensorflow.keras.applications import Xception  # Added import
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.regularizers import l2

class EnhancedDetector:
    def __init__(self, model_path='models/deepfake_xception.weights.h5'):
        self.input_size = (299, 299)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.model = self.build_model()
        self.model.load_weights(model_path)

    def build_model(self):
        # Fixed syntax and added proper input_shape
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(299, 299, 3)  # Fixed parameter
        )
        
        # Architecture matches training
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=base_model.input, outputs=predictions)  # Fixed syntax

    def preprocess(self, face_img):
        img = cv2.resize(face_img, self.input_size)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    def predict(self, image_path, confidence_threshold=0.65):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return "No face detected", 0.0
        
        x,y,w,h = faces[0]
        face = img[y:y+h, x:x+w]
        
        processed = self.preprocess(face)
        confidence = float(self.model.predict(processed)[0][0])
        
        label = "Fake" if confidence > confidence_threshold else "Real"
        return label, confidence
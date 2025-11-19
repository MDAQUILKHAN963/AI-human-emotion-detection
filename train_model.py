import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load data
df = pd.read_csv('fer2013.csv')

def preprocess():
    pixels = np.array([np.fromstring(image, sep=' ') for image in df['pixels']])
    X = pixels.reshape((pixels.shape[0], 48, 48, 1)).astype('float32') / 255.
    y = to_categorical(df['emotion'], num_classes=7)
    X_train, X_test = X[df['Usage']=='Training'], X[df['Usage']!='Training']
    y_train, y_test = y[df['Usage']=='Training'], y[df['Usage']!='Training']
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = preprocess()

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))
model.save('emotion_model.h5')
print("Model trained and saved as emotion_model.h5")
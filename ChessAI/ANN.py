from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

A = np.arange(42)
B = np.reshape(A, (-1, 6))

model = Sequential()
model.add(Dense(12, input_shape=(64,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset

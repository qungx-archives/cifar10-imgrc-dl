import tensorflow as tf
import keras
from keras import datasets
from keras.datasets import cifar10
from keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, Flatten
import pathlib
from pathlib import Path

(x_train, y_train) , (x_test, y_test) = cifar10.load_data()

# Load dts to 0-to-1 range
x_train = x_train.astype("float32")  
x_test = x_test.astype("float32")
x_test /=255
x_train /= 255

# Convert to binary class matrices
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a model and add layers
model = keras.Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights.h5")

# model.summary()






import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load dtset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize dts to 0-to-1 range
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a model and add layers
model = tf.keras.Sequential()
# Adding convolutional layer (2D layer, 32 different filters, 3x3 tiles, )
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
# To switch between Convolutional layer and Dense Layer, we need a Flatten layer
model.add(Flatten())
# Hidden Layers:  512 nodes (neurons), activation funct: relu, input: 32 pixels by 32 pixels and 3 colors (RGB)
model.add(Dense(512, activation="relu")) 
# Output Layer: 10 nodes (neurons), activation funct: softmax
model.add(Dense(10, activation="softmax"))

# Print a summary of the model
model.summary()
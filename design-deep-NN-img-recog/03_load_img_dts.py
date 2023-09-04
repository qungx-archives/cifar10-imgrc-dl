import tensorflow as tf
from tensorflow import keras

# Load dts
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize dts set to 0-to-1 range
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train / 255
x_test = x_test / 255

# Convert class vectors to binary class matrices 
# Labels from 0 to 9
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)



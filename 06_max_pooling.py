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
x_test /=255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a model and add layers
model = tf.keras.Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(32, 32, 3),  activation="relu"))
model.add(Conv2D(32, (3,3), activation="relu"))
# Keeping the more important zone and then remove the less important one
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()
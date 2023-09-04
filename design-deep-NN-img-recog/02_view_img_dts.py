import tensorflow as tf
import matplotlib.pyplot as plt

cifar10_class_names = {
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Boat",
    9: "Truck"
}

(x_train, y_train), (x_test, y_test) = tf.keras.cifar10.load_data()

# Loop through each picture in the data set
for i in range(1000):
    # Grab an image from the data set
    smpl_img = x_train[i]
    # Grab the image's expected class id
    img_class_num = y_train[i][0]
    # Look up the class name from class id
    img_class_name = cifar10_class_names[img_class_num]

    #Draw the img as a plot
    plt.imshow(smpl_img)
    #Label the image
    plt.title(img_class_name)
    #Show the plot on the screen
    plt.show()
    




########### Basic Parameters for Running: ################################

TFliteNamingAndVersion = "dig_bk"     # Used for tflite Filename
Training_Percentage = 0.0             # 0.0 = Use all Images for Training
Epoch_Anz = 150
Model_Version = 0
Show_Plots = False

##########################################################################


import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from sklearn.utils import shuffle
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import to_categorical
from PIL import Image
from pathlib import Path
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

loss_ges = np.array([])
val_loss_ges = np.array([])

# %matplotlib inline
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# Load training data
Input_dir = 'ziffer_sortiert_resize'

files = glob.glob(Input_dir + '/*.jpg')
x_data = []
y_data = []

for aktfile in files:
    base = os.path.basename(aktfile)
    target = base[0:1]
    if target == "N":
        category = 10                # NaN does not work --> convert to 10
    else:
        category = int(target)
    test_image = Image.open(aktfile)
    test_image = np.array(test_image, dtype="float32")
    x_data.append(test_image)
    y_data.append(np.array([category]))

x_data = np.array(x_data)
y_data = np.array(y_data)
y_data = to_categorical(y_data, 11)
print(x_data.shape)
print(y_data.shape)

x_data, y_data = shuffle(x_data, y_data)

if (Training_Percentage > 0):
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=Training_Percentage)
else:
    X_train = x_data
    y_train = y_data


# Define the model
model = Sequential()
if Model_Version == 0:
    model.add(BatchNormalization(input_shape=(32, 20, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(11, activation="softmax"))
elif Model_Version == 1:
    model.add(BatchNormalization(input_shape=(32, 20, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(11, activation="softmax"))
elif Model_Version == 2:
    model.add(BatchNormalization(input_shape=(32, 20, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(11, activation="softmax"))
elif Model_Version == 3:
    model.add(BatchNormalization(input_shape=(32, 20, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(11, activation="softmax"))


model.summary()

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
    metrics=["accuracy"])


# Training
Batch_Size = 4
Shift_Range = 1
Brightness_Range = 0.3
Rotation_Angle = 10
ZoomRange = 0.4

datagen = ImageDataGenerator(
    width_shift_range=[-Shift_Range, Shift_Range],
    height_shift_range=[-Shift_Range, Shift_Range],
    brightness_range=[1 - Brightness_Range, 1 + Brightness_Range],
    zoom_range=[1 - ZoomRange, 1 + ZoomRange],
    rotation_range=Rotation_Angle)

if (Training_Percentage > 0):
    train_iterator = datagen.flow(x_data, y_data, batch_size=Batch_Size)
    validation_iterator = datagen.flow(X_test, y_test, batch_size=Batch_Size)
    history = model.fit(
        train_iterator, validation_data=validation_iterator, epochs=Epoch_Anz)
else:
    train_iterator = datagen.flow(x_data, y_data, batch_size=Batch_Size)
    history = model.fit(train_iterator, epochs=Epoch_Anz)


# Plot
loss_ges = np.append(loss_ges, history.history['loss'])
plt.semilogy(history.history['loss'])

if (Training_Percentage > 0):
    val_loss_ges = np.append(val_loss_ges, history.history['val_loss'])
    plt.semilogy(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'eval'], loc='upper left')
if Show_Plots:
    plt.show()


# Check each image for expected and deviation
Input_dir = 'ziffer_sortiert_resize'
res = []
only_deviation = True
show_wrong_image = True

files = glob.glob(Input_dir + '/*.jpg')

for aktfile in files:
    base = os.path.basename(aktfile)
    target = base[0:1]
    if target == "N":
        zw1 = -1
    else:
        zw1 = int(target)
    expected_class = zw1
    image_in = Image.open(aktfile)
    test_image = np.array(image_in, dtype="float32")
    img = np.reshape(test_image, [1, 32, 20, 3])
    classes = np.argmax(model.predict(img), axis=-1)
    classes = classes[0]
    if classes == 10:
        classes = -1
    zw2 = classes
    zw3 = zw2 - zw1
    res.append(np.array([zw1, zw2, zw3]))
    if only_deviation == True:
        if str(classes) != str(expected_class):
            print(aktfile + " " + str(expected_class) + " " + str(classes))
            if show_wrong_image == True:
                display(image_in)
    else:
        print(aktfile + " " + aktsubdir + " " + str(classes))


res = np.asarray(res)


plt.plot(res[:, 0])
plt.plot(res[:, 1])
plt.title('Result')
plt.ylabel('Digital Value')
plt.xlabel('#Picture')
plt.legend(['real', 'model'], loc='upper left')
if Show_Plots:
    plt.show()


# Save the model¶
FileName = TFliteNamingAndVersion + "_v" + str(Model_Version)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(FileName + ".tflite", "wb").write(tflite_model)

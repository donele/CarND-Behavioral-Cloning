import csv
import matplotlib.pyplot as plt
import numpy as np

# Read csv file created by simulator.
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
sideCamCorr = 0.15
doFlip = False
for line in lines:
    # Read three images from the center, left, and right cameras.
    img_paths = line[0:3]
    steering = float(line[3])
    img3 = [plt.imread(path) for path in img_paths]
    meas3 = [steering, steering + sideCamCorr, steering - sideCamCorr]

    # Flip every other images, with some randomness.
    if np.random.binomial(1, 0.9):
        doFlip = not doFlip
    if doFlip:
        img3 = [np.fliplr(img) for img in img3]
        meas3 = [-x for x in meas3]

    images.extend(img3)
    measurements.extend(meas3)

# Begin training and validation.
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(12,5,5,activation='relu',subsample=(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(12,5,5,activation='relu',subsample=(3,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(160, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, batch_size=128)
model.save('model.h5')

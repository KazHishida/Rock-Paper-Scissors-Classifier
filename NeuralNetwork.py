from tensorflow import keras
from keras import preprocessing
import numpy as np
import os
import cv2
import random

classifications = ['paper', 'rock', 'scissors']
typetonum = {'paper':0,'rock':1,'scissors':2}
PATH = os.getcwd()

train_path = PATH + '/data/'
train_data = os.listdir(train_path)
x_train = []
y_train = []
x_test = []
y_test = []
order = []
for x in range(len(train_data)):
    order.append(x)
random.shuffle(order)

for filename in order:
    img_path = train_path+train_data[(filename)]
    x=preprocessing.image.load_img(img_path)
    x = cv2.cvtColor(preprocessing.image.img_to_array(x), cv2.COLOR_BGR2GRAY) #GRAYSCALE
    x = cv2.resize(preprocessing.image.img_to_array(x), dsize=(275, 275), interpolation=cv2.INTER_CUBIC) #COMPRESSION
    for type in classifications:
        if type in train_data[(filename)]:
            y_train.append(typetonum[type])
    x = preprocessing.image.img_to_array(x) / 255.0
    x_train.append(x)

x_test = x_train[int(len(train_data)//1.15):]
y_test = y_train[int(len(train_data)//1.15):]
x_train = x_train[:int(len(train_data)//1.15)]
y_train = y_train[:int(len(train_data)//1.15)]

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (275,275,1)), #175/175/3 for colored compressed
    keras.layers.Dense(3600, activation = 'relu'),
    keras.layers.Dense(3, activation = 'softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(np.asarray(x_train), np.asarray(y_train), epochs=9)

model_loss, model_acc = model.evaluate(np.asarray(x_test), np.asarray(y_test))

print(f"Test Accuracy: {model_acc}\nTest Loss: {model_loss}")

if model_acc>.9:
    print("Successful Test. Saving model")
    model.save("handNeuralNetwork.h5")
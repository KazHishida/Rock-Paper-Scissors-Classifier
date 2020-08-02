#FOR TESTING INDIVIDUAL PICTURES, PUT THEM IN MYHAND FOLDER
from tensorflow import keras
from keras import preprocessing
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

classifications = ['paper', 'rock', 'scissors']
typetonum = {'paper':0,'rock':1,'scissors':2}
model = keras.models.load_model('handNeuralNetwork.h5')

PATH = os.getcwd()

hand_path = PATH + '/MYHAND/'
hand_data = os.listdir(hand_path)
for filename in hand_data:
    img_path = hand_path+filename
    x=preprocessing.image.load_img(img_path)
myhands = []
for filename in hand_data:
    img_path = hand_path + filename
    x = preprocessing.image.load_img(img_path)
    x = cv2.cvtColor(preprocessing.image.img_to_array(x), cv2.COLOR_BGR2GRAY)
    x = cv2.resize(preprocessing.image.img_to_array(x), dsize=(275, 275), interpolation=cv2.INTER_CUBIC)
    x = preprocessing.image.img_to_array(x) / 255.0
    myhands.append(x)
predictions2 = model.predict(np.asarray(myhands))
plt.figure(figsize=(5,5))
for i in range(len(myhands)):
    plt.grid(False)
    myhands[i] = np.reshape(myhands[i], (275, 275))
    myhands[i] = Image.fromarray(np.uint8(myhands[i] * 255), 'L')
    plt.imshow(myhands[i], cmap=plt.cm.binary)
    plt.title(classifications[np.argmax(predictions2[i])])
    plt.show()

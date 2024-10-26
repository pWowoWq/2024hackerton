import tensorflow as tf
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Add
import numpy as np
import pandas as pd
import time
##------------------데이터 받아오기----------------------------##
data_dir = 'C:\Users\user\Desktop\2024.10.26\data'
folders=[]
folders = [f for f in (os.listdir(data_dir))]
print(folders)

input_height = 48
input_width = 48
input_channel = 8

input_shape = (input_height, input_width, input_channel)
num_classes = len(folders) - 1  # number of folders (minus the 'test' folder)

trainset = []
testset = []
label = []
test_files = []

for i in range(len(folders)):
    
    print("Loading images: " + folders[i])
    data_path = os.path.join(data_dir, str(folders[i]))
        
    for subdir, dirs, files in os.walk(data_path):
        for filename in files:
            file_path = data_path + os.sep + filename

            if file_path.endswith(".jpg") or file_path.endswith(".png"):
                image = cv2.imread(file_path)
                resized_image = cv2.resize(image,(input_width, input_height))

                if folders[i] == "test":
                    testset.append(resized_image)
                    test_files.append(file_path)

                else: 
                    trainset.append(resized_image)
                    
                    label_data = np.zeros(shape = (num_classes)) 
                    label_data[i] = 1.0
                    label.append(label_data)           

trainset = np.array(trainset)
testset = np.array(testset)
label = np.array(label)
print("# of Training Images: ", trainset.shape[0])
print("# of Test Images: ", testset.shape[0])

##------------------데이터 전처리------------------------------##

##------------------모델--------------------------------------##
def Model():
  model = Sequential()
  model.add(Convolution2D(24,(5,5),strides=(2,2),input_shape=(66,200,3),activation="elu"))
  model.add(Convolution2D(36,(5,5),strides=(2,2),activation="elu"))
  model.add(Convolution2D(48,(5,5),strides=(2,2),activation="elu"))
  model.add(Convolution2D(64,(3,3),activation="elu"))
  model.add(Convolution2D(64,(3,3),activation="elu"))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100,activation="elu"))
  model.add(Dropout(0.5))
  model.add(Dense(50,activation="elu"))
  model.add(Dropout(0.5))
  model.add(Dense(10,activation="elu"))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.compile(optimizer=Adam(learning_rate=1e-3),loss="mse")
  return model
model = Model()
model.summary()
##--------------------학습----------------------------------------------##
##--------------------모델 저장-----------------------------------------##
model.save('model.h5')
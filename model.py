import numpy as np
import cv2
import os
import json
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Add
from tensorflow.keras.optimizers import Adam
##-----------------------------------------------------##
folders=[]
data_dir = os.path.join('C:/Users/user/Desktop/2024.10.26/data')
##-------------------------------------------------------##
input_height = 360
input_width = 540
input_channel = 2
input_shape = (input_height, input_width, input_channel)
##------------------------------------------------------##
trainset = []
testset = []
label = []
test_files = []
images = []
##----------------------------##

data_path= os.path.join(str('C:\\'),'Users','user','Desktop','2024.10.26','data')
data_path_test = os.path.join(data_path,str('test'))
data_path_train = os.path.join(data_path, str('train'))
train_folders = [f for f in (os.listdir(data_path_train))]
num_classes = len(train_folders)
for subdir, dirs, files in os.walk(data_path_test):
    for filename in files:
        test_data_path = os.path.join(data_path_test,filename)
        if test_data_path.endswith(".jpg"):
            image1 = cv2.imread(test_data_path)
            testset.append(image1)
for i, folder in enumerate(train_folders):
    train_data_path = os.path.join(data_path_train, folder)
    for filename in os.listdir(train_data_path):
        if filename.endswith(".jpg"):
            file_path = os.path.join(train_data_path, filename)
            image = cv2.imread(file_path)
            trainset.append(image)
            label_data = np.zeros(shape=(num_classes,))
            label_data[i] = 1.0
            label.append(label_data)
##---------------------------------------------------------------##       
trainset = np.array(trainset)
testset = np.array(testset)
label = np.array(label)
print("# of Training Images: ", trainset.shape[0])
print("# of Test Images: ", testset.shape[0])
##-------------------------------------------------##
trainset = trainset/225.0
testset = testset/225.0
##-------------------------------------------------##
input_height = 360
input_width = 540
input_channel = 3
input_shape = (input_height, input_width, input_channel)

def Model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding="valid", input_shape=(288, 352, 3)))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='valid'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
model = Model()
model.summary
##------------------------------------------------------------##
adam = Adam()
model.compile(
    optimizer = adam
    ,loss = 'categorical_crossentropy'
    ,metrics=['accuracy']
)
history = model.fit(
    trainset,label,
    batch_size = 20,
    epochs = 5,
    validation_split=0.005
)
model_desc = model.to_json()
with open('./Model/model.json', 'w') as file_model:
    file_model.write(model_desc)
model.save_weights('./Model/model.weights.h5')

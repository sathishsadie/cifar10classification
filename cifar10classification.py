import numpy as np 
import pandas as pd
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
x_tr=np.load('x_train.npy')
y_tr=np.load('y_train.npy')
x_te=np.load('x_test.npy')
y_te=np.load('y_test.npy')
stop=EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=0,
    verbose=1,
    baseline=0.05
)
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=model.fit(x_tr,y_tr,validation_data=(x_te,y_te),epochs=100,callbacks=[stop])


plt.plot(history.history['accuracy'],label='accuracy',c='b')
plt.plot(history.history['val_accuracy'],label='val_accuracy',c='r')
plt.plot(history.history['loss'],label='loss',c='g')
plt.plot(history.history['val_loss'],label='Val loss',c='orange')
plt.axis('off')
plt.legend()
plt.show()
model.save('cifar.h5')
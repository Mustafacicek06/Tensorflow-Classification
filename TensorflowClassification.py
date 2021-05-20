# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:03:30 2020

@author: GRUNDİG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

dataFrame = pd.read_excel("maliciousornot.xlsx")

dataFrame.info()

dataFrame.describe()

# ilişki 
dataFrame.corr()["Type"].sort_values()

# iki tarafta dengeli olması önemli 
# bi taraf cok az ise eğitilemeyebilir

sbn.countplot(x="Type",data=dataFrame)

dataFrame.corr()["Type"].sort_values().plot(kind="bar")

# numpy dizisine çevirdik
y = dataFrame["Type"].values

x = dataFrame.drop("Type",axis=1).values

from sklearn.model_selection import train_test_split

x_train , x_test, y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=15)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train= scaler.transform(x_train)

x_test = scaler.transform(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Dropout ve EarlyStopping 'i overfitting problemlerini çözmek için kullanırız

# (383, 30)
x_train.shape


model = Sequential()
# kaç tane kolon varsa nöronu ona göre ayarlamak önerilir
model.add(Dense(units=30,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=15,activation="relu"))
# sigmoid (0,1) arasında değer verir 
# output layerinize koyarak bunu elde edebilirsiniz
model.add(Dense(units=1,activation="sigmoid"))

# Sınıflandırma fonksiyonu olduğu için mse değil binary_crossentropy
model.compile(loss="binary_crossentropy",optimizer="adam")

model.fit(x=x_train,y=y_train,epochs=750,validation_data=(x_test,y_test),verbose=1)

lossData = pd.DataFrame(model.history.history)

# kendi içinde tutarlı olmaya çalışıyor overfitting oluyor
# yeni bir veri verdiğimiz zaman predicti çok saçma şeyler olacak
lossData.plot()


model = Sequential()
# kaç tane kolon varsa nöronu ona göre ayarlamak önerilir
model.add(Dense(units=30,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=15,activation="relu"))
# sigmoid (0,1) arasında değer verir 
# output layerinize koyarak bunu elde edebilirsiniz
model.add(Dense(units=1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam")

# tahammül = patience 
earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)

model.fit(x=x_train, y=y_train,epochs=700,validation_data=(x_test,y_test),verbose=1,callbacks=[earlyStopping])    


modelKaybi= pd.DataFrame(model.history.history)
modelKaybi.plot()


## Dropout yani katman ve nöronlarda fazlalık varsa optimize ayarı

model = Sequential()
# kaç tane kolon varsa nöronu ona göre ayarlamak önerilir
# Dropoutu 0.5 den büyük yapmamaya dikkat etmemiz gerek
model.add(Dense(units=30,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=15,activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=15,activation="relu"))
model.add(Dropout(0.6))

# sigmoid (0,1) arasında değer verir 
# output layerinize koyarak bunu elde edebilirsiniz
model.add(Dense(units=1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam")

model.fit(x=x_train, y=y_train,epochs=700,validation_data=(x_test,y_test),verbose=1,callbacks=[earlyStopping])    


modelKaybi= pd.DataFrame(model.history.history)
modelKaybi.plot()

tahminlerimiz = model.predict_classes(x_test)

from sklearn.metrics import classification_report,confusion_matrix

#  0 ile 1 leri ne kadar doğru tahmin etmiş ona bakıyoruz
print(classification_report(y_test,tahminlerimiz))

print(confusion_matrix(y_test,tahminlerimiz))







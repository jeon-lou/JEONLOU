#!/usr/bin/env python
# coding: utf-8

# In[8]:


###########         auto clave model   ##################

import random

def calc_pum(temp, pressure, time):
    if temp >  60 or pressure > 7 or time > 10: return "bad"
    elif temp >  40 or pressure > 4 or time > 6 : return "normal"
    return "good"

# psa = h, pressure = w
fp = open("e:/data/psa/autoclave.csv","w",encoding="utf-8")
fp.write("temp,pressure,time,label\r\n")
cnt = {"good":0, "normal":0, "bad":0}
for i in range(200):
    temp = random.randint(20,70)
    pressure = random.randint(2,10)
    time = random.randint(2,15)
    label = calc_pum(temp, pressure, time)
    cnt[label] += 1
    fp.write("{0},{1},{2},{3}\r\n".format(temp, pressure, time, label))
fp.close()
print("data completed,", cnt)


# In[9]:


from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tbl = pd.read_csv("e:/data/psa/autoclave.csv")

label = tbl["label"]
temp = tbl["temp"] / 50
pressure = tbl["pressure"] / 6
time = tbl["time"] / 8

wh = pd.concat([temp,pressure,time], axis=1)

data_train, data_test, label_train, label_test =     train_test_split(wh, label)

clf = svm.SVC()
clf.fit(data_train, label_train)
#predict = clf.predict(data_test)


# In[11]:


raw = [30/50, 4/6, 7.5/8]
test = np.array(raw)
test = test.reshape(1,-1)
print(clf.predict(test))


# In[ ]:





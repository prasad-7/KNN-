import sklearn
from sklearn.utils import shuffle
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing

neighbours = int(input("Neighbours for classification >> "))

data = pd.read_csv("car.data")
#print(data.head())

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
safety = le.fit_transform(list(data["safety"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
cls = le.fit_transform(list(data["class"]))
door = le.fit_transform(list(data["door"]))

x = list(zip(buying,maint,safety,persons,lug_boot,door))
y = list(cls)

x_train , x_test , y_train , y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

model = KNeighborsClassifier(n_neighbors=neighbours)


model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
p= model.predict(x_test)

names = ["unacc", 'acc', 'good','vgood']

for v in range(len(p)):
    print("Predicted : ", names[p[v]], "Data : ", x_test[v], "Actual : ", names[y_test[v]])

print("Accuracy of the model >> ", acc)
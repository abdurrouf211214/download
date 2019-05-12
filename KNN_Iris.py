import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.linear_model import logistic_regression_path
data_iris=datasets.load_iris()
print(data_iris.data)
print(data_iris.target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(data_iris.data,data_iris.target,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)

from sklearn.metrics import accuracy_score
print("accuracy is = ", accuracy_score(y_test,clf.predict(x_test)))

accuracy_values=[]
for x in range(1,x_train.shape[0]):
    clf=KNeighborsClassifier(n_neighbors=x).fit(x_train,y_train)
    accuracy=accuracy_score(y_test,clf.predict(x_test))
    accuracy_values.append([x,accuracy])
    pass
accuracy_values=np.array(accuracy_values)

plt.plot(accuracy_values[:,0],accuracy_values[:,1])
plt.xlabel("K")
plt.ylabel("accuracy")
plt.show()

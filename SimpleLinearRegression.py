           
#libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#uploading dataset file
uploadFile = files.upload()
dataset = pd.read_csv("Salary_Data.csv")

#divide dataset into x and y
x = dataset.iloc[:,:-1].values
# u can write x = dataset.iloc[:,0]
y = dataset.iloc[:,1].values

#determine the test size and training size
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)

#fit the model
model = LinearRegression()
model.fit(x_train, y_train)
result = model.predict(x_test)


#show the result as training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train,model.predict(x_train), color = 'blue')
plt.title("Salary VS Expereince (Training Set)")
plt.xlabel("Expereince Years")
plt.ylabel("Salary")
plt.show()

#show the result as test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,model.predict(x_train), color = 'blue')
plt.title("Salary VS Expereince (Test Set)")
plt.xlabel("Expereince Years")
plt.ylabel("Salary")
plt.show()


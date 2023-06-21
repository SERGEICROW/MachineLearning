import numpy as np
import pandas as pd
import tensorflow as tf

#Importing dataset
dataset = pd.read_csv('/home/roy/Projects/Learning/ML1/data/Churn_Modelling.csv')
#Creates matrix column x taking all the columns starting from number 3 and not the last one (First is datarow, 2 and 3 are irrelevant data, last one is our expected result)
x = dataset.iloc[:, 3:-1].values
#Create matrix column y taking only the expected result
y = dataset.iloc[:, -1].values

#Importing library to encode a column
from sklearn.preprocessing import LabelEncoder
#Assigning library to constant
le = LabelEncoder()
#Calling all the rows with X indexes and chosing column 3 wich has index 2 and encoding(assigning an especific number to a value) in this case for column 3
x[:, 2] = le.fit_transform(x[:, 2])

#Importing library to enconde and transform column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Importing library to split a dataset for training and test with a library
from sklearn.model_selection import train_test_split
#creates an entity for each matriz (X and Y) using the train_test_split library
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Importing library to scale data wich is necearry for machine learning
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Initializing the ann by creating a neural network as an object from a tensorflow class
ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
#adding to the ann the module to add layers which is from the keras module of tensorflow
#adding Dense class which takes arguments
#experiment with any amount but 6 is average.
#Units = number of hidden neurons
#activation = name of activation function of the hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Adding the output layer which contains the number of outputs, in this case we need 1 neuron because we are getting binary outputs (0 or 1)
#Adding the activation function sigmoid
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Compiling the ANN
#Optimizer iterates to reduce errors
#Loss chosed binary for binary outputs like this example
#Metrics to metric ann evaluation
ann.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])

#Training the ann on the training set
#When training we need other 2 parameters batch_size(to compare predictions in batch)
#Last parameter for training is epochs
ann.fit(x_train, y_train, batch_size=32, epochs=100)

#Making the ann predict if 1 especific customer will leave the bank, the outcome comes in probability
#Any predict method must be within an array (2d array)
#We are inserting encoded data from 1 row
print("------------------Printing percentages------------------")
print(ann.predict(sc.transform([[1, 0, 0 , 600, 1, 40, 3, 6000, 2, 1, 1, 50000]])))
#To return a number (1 or 0) we need to add a condition
print("--------------------Printing Values---------------------")
print(ann.predict(sc.transform([[1, 0, 0 , 600, 1, 40, 3, 6000, 2, 1, 1, 50000]])) > 0.5)

#Predicting the test set results
print("-----------------Printing Test set results -------------")
print("[reality vs prediction]")
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making the confusion Matrix
print("-----------------Printing COnfusion Matrix--------------")
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MAHALAKSHMI R
RegisterNumber:  212223230116
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```

## Output:
##HEAD(),INFO() & NULL():
![Screenshot 2024-09-27 132322](https://github.com/user-attachments/assets/7c5a5ab3-230c-4a69-8098-4b0780ec5d4f)

##Converting string literals to numerical values using label encoder:
![Screenshot 2024-09-27 132334](https://github.com/user-attachments/assets/59953a99-7e6c-4497-945c-10a2c8ca8091)

##MEAN SQUARED ERROR:
![Screenshot 2024-09-27 132341](https://github.com/user-attachments/assets/4728c74b-06ec-41fd-a581-711f08a98892)

##R2 (Variance):
![Screenshot 2024-09-27 132349](https://github.com/user-attachments/assets/b0c12dd0-b1d0-4a40-b077-366fdaee59b6)

##DATA PREDICTION & DECISION TREE REGRESSOR FOR PREDICTING THE SALARY OF THE EMPLOYEE:
![Screenshot 2024-09-27 132403](https://github.com/user-attachments/assets/a5bcd986-adaa-4688-99db-79a2957ffdc1)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

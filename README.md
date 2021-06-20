# spark-foundation-task-1
#Importing required libraries

#To work with data frame
import pandas as pd
#To perform numerical operations
import numpy as np
#For plotting graphs
import matplotlib.pyplot as plt
%matplotlib inline
#For running regression 
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



#Reading the dataset

url="https://bit.ly/w-data"
data=pd.read_csv(url)



#Exploring data

print(data.shape)
data.head()
data.describe()
data.info()



#Graphical view of data

data.plot(x="Hours",y="Scores",kind="scatter")
plt.title('Hrs Studied vs Percentage Score')
plt.xlabel('Hours student studied')
plt.ylabel('% of student\'s score')
plt.show()



#Finding correlation between two variables

data.corr(method="pearson")
data.corr(method="spearman")

hours=data["Hours"]
scores=data["Scores"]

x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
x



#Devide the data into train and test dataset

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

reg=LinearRegression(fit_intercept=True)
reg.fit(x_train,y_train)
print("Training complete......")



#Finding linear regression

#Plotting the regression line 
m=reg.coef_
c=reg.intercept_
line=m*x+c

#Plotting for the test data
plt.scatter(x,y)
plt.plot(x,line,color='y');
plt.show()

#Prediction making
print(x_test)
y_predict=reg.predict(x_test)



#Comparison between Actual value and Predicted value

df=pd.DataFrame({'Actual':y_test,'Predicted':y_predict})
df



#Final visualization

df.plot(kind='line')
df.plot(kind='bar')
plt.grid(which='major',linewidth='0.5',color='red')
plt.grid(which='minor',linewidth='0.5',color='blue')
plt.title('Hrs Studied vs Percentage Score')
plt.xlabel('Hours student studied')
plt.ylabel('% of student\'s score')
plt.show()



#Estimating training and test score

print("Training score:",reg.score(x_train,y_train))
print("Test score:",reg.score(x_test,y_test))



#Predicted score if a student studies for 9.25hrs/day

hrs=[9.25]
ans=reg.predict([hrs])
print("Hours student study={}".format(hrs))
print("Predicted score of student={}".format(ans[0]))



#Finding the residuald

from sklearn import metrics
print ("Mean Absolute Error=>",metrics.mean_absolute_error(y_test,y_predict))
print("Mean Square Error=>",metrics.mean_squared_error(y_test,y_predict))
print("Root Mean Squared Error=>",np.sqrt(metrics.mean_squared_error(y_test,y_predict)))



#RESULT

an approx 93% is achive by student if he/she studies for 9.25 hrs/day

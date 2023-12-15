from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mtl
import joblib



# Load the feature dataset
feature_data = pd.DataFrame(pd.read_excel('planetrydataset.xlsx'))




# # Drop non-numeric and irrelevant columns
X=feature_data.drop(columns=['name','target'])
Y=feature_data['target']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# clf=RandomForestClassifier()
# clf.fit(X_train,Y_train)
# print(clf.feature_importances_)
# print(clf.predict(X_test))
# print(clf.predict([[13.8,1,1,804.10485,13.215704]]))
# print(clf.score(X_test,Y_test))
# joblib.dump(clf,'./model.job.lib',compress=3)

clf1=LinearRegression()
clf1.fit(X_train,Y_train)
print(clf1.predict(X_test))
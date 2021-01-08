import streamlit as st
st.title('Diabetes Prediction App')
a=st.number_input("Pregnancies")

b=st.number_input("Glucose")
c=st.number_input("BloodPressure")
d=st.number_input("Skin_Thickness")
e=st.number_input("Insulin")
f=st.number_input("BMI")
g=st.number_input("Diabetes_Pedigree_Function")
h=st.number_input("Age")
import numpy as np , pandas as pd, sklearn
df=pd.read_csv("diabetes.csv")
x=df.drop(["Outcome"],axis=1)
x=x.iloc[:,:].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction=model.predict([[a,b,c,d,e,f,g,h]])
button_pressed=st.button("Predict")
if button_pressed:
    if prediction[0]==0:

        st.write("You don't have diabetes")
    else:
        st.write("You have diabetes")


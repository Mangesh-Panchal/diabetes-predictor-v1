import streamlit as st
st.title('Diabetes Prediction App')
a=st.text_input("Pregnancies")

b=st.slider("Glucose",0.0,0.1)
c=st.slider("BloodPressure",0,100)
d=st.slider("Skin_Thickness",0,100)
e=st.slider("Insulin",0,100)
f=st.slider("BMI",0,100)
g=st.slider("Diabetes_Pedigree_Function",0,100)
h=st.slider("Age",0,200)
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
prediction=model.predict([[float(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h)]])
button_pressed=st.button("Predict")
if button_pressed:
    st.write(a)

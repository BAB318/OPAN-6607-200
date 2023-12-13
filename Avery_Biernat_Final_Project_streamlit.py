# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#load packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

#load data
s = pd.read_csv('social_media_usage.csv')

#create function to set var equal to 1 or 0
def clean_sm(x):
    result = np.where(x == 1, 1, 0)
    return result


#creates clean df
#creates target var
ss = s
ss['sm_li'] = clean_sm(ss['web1h'])
ss = ss[['income', 'educ2', 'par', 'marital', 'gender', 'age', 'sm_li']]

#sets na values as specified in assignment/documents
ss['income'] = np.where(ss['income'] > 9, np.NaN, ss['income'])
ss['educ2'] = np.where(ss['educ2'] > 8, np.NaN, ss['educ2'])
ss['par'] = np.where(ss['par'] > 2, np.NaN, ss['par'])
ss['gender'] = np.where(ss['gender'] > 3, np.NaN, ss['gender'])
ss['age'] = np.where(ss['age'] > 98, np.NaN, ss['age'])


#drops na. Sets variables to specified values
ss = ss.dropna()
ss['par'] = clean_sm(ss['par'])
ss['marital'] = clean_sm(ss['marital'])
ss['gender'] = np.where(ss['gender'] == 2, 1, 0)

#creates Target/Features
y = ss['sm_li']
x = ss[['income', 'educ2', 'par', 'marital', 'gender', 'age']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#fits Logistic Regression
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(x_train, y_train)


#creates datset for prediction
prediction_data = pd.DataFrame({
    'income': [0],
    'educ2': [0],
    'par': [0],
    'marital': [0],
    'gender': [0],
    'age': [0],
})

#streamlit app
st.header("Avery Biernat - Final Project")
st.subheader("Prediction of LinkedIn Users Through Logistic Regression")

#Checks prediction values

gender_status = st.radio("Select Gender: ", ('Male', 'Female'))

if (gender_status  == 'Female'):
    prediction_data['gender'] = 1
else:
    prediction_data['gender'] = 0

maritial_status = st.radio("Select Maritial Status: ", ('Single', 'Married'))
if (maritial_status  == 'Married'):
    prediction_data['marital'] = 1
else:
    prediction_data['marital'] = 0
    
parental_status = st.radio("Select Parental Status: ", ('Parent', 'Non-Parent'))
if (maritial_status  == 'Parent'):
    prediction_data['par'] = 1
else:
    prediction_data['par'] = 0

education_status = st.selectbox("Education: ",
                     ['Less than high school', 'High school, incomplete', 'High school, graduated', 'Some college, no degree', 'Two-year associate degree', 'Four-year Bachelor’s degree', 'Postgraduate or professional schooling, no postgraduate degree', 'Postgraduate or professional degree'])
if (education_status == 'Less than high school'):
    prediction_data['educ2'] = 1
elif (education_status == 'High school, incomplete'):
    prediction_data['educ2'] = 2
elif (education_status == 'High school, graduated'):
    prediction_data['educ2'] = 3
elif (education_status == 'Some college, no degree'):
    prediction_data['educ2'] = 4
elif (education_status == 'Two-year associate degree'):
    prediction_data['educ2'] = 5
elif (education_status == 'Four-year Bachelor’s degree'):
    prediction_data['educ2'] = 6
elif (education_status == 'Postgraduate or professional schooling, no postgraduate degree'):
    prediction_data['educ2'] = 7
elif (education_status == 'Postgraduate or professional degree'):
    prediction_data['educ2'] = 8

income_status = st.selectbox("Income: ",
                     ['Less than $10,000', '10 to under $20,000', '20 to under $30,000', '30 to under $40,000', '40 to under $50,000', '50 to under $75,000', '75 to under $100,000', '100 to under $150,000', '$150,000 or more'])
if (income_status == 'Less than $10,000'):
    prediction_data['income'] = 1
elif (income_status == '10 to under $20,000'):
    prediction_data['income'] = 2
elif (income_status == '20 to under $30,000'):
    prediction_data['income'] = 3
elif (income_status == '30 to under $40,000'):
    prediction_data['income'] = 4
elif (income_status == '40 to under $50,000'):
    prediction_data['income'] = 5
elif (income_status == '50 to under $75,000'):
    prediction_data['income'] = 6
elif (income_status == '75 to under $100,000'):
    prediction_data['income'] = 7
elif (income_status == '100 to under $150,000'):
    prediction_data['income'] = 8
elif (income_status == '$150,000 or more'):
    prediction_data['income'] = 9

age_status = st.slider("Age", 18, 99, 18)
prediction_data['age'] = age_status

predicted_value = logreg.predict(prediction_data)
predicted_prob = logreg.predict_proba(prediction_data)

st.write(f"Linkedin User Prediction: {predicted_value}")
st.write(f"Linkedin User Prediction Probability: {predicted_prob}")




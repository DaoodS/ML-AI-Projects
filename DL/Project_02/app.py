import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gen = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    standscaler = pickle.load(file)

st.title('Churn Prediction')

# User Input
geography = st.selectbox('Geography', onehot_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gen.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit = st.selectbox('Credit Card', [0, 1])
is_active = st.selectbox('Active member', [0, 1])

# Input Data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_geo.transform(input_data['Geography']).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = standscaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

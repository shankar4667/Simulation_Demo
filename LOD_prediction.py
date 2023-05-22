#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate random sample data for the variables
np.random.seed(42)  # For reproducibility

num_samples = 1000

air_velocity = np.random.uniform(low=0.5, high=2.5, size=num_samples)
steam_inlet_temp = np.random.uniform(low=100, high=200, size=num_samples)
steam_outlet_temp = np.random.uniform(low=200, high=300, size=num_samples)
steam_pressure = np.random.uniform(low=1, high=10, size=num_samples)
ambient_temp = np.random.uniform(low=20, high=30, size=num_samples)
moisture_content = np.random.uniform(low=0, high=1, size=num_samples)

# Combine the variables into a feature matrix
X = np.column_stack((air_velocity, steam_inlet_temp, steam_outlet_temp, steam_pressure, ambient_temp))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, moisture_content, test_size=0.2, random_state=42)

# Fit a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Get the coefficients and intercept of the linear regression model
coefficients = regressor.coef_
intercept = regressor.intercept_

# Predict moisture content for all sample data points
predicted_moisture_content = regressor.predict(X)

# Print the predicted equation
equation = "Moisture Content = {:.5f} + ({:.5f} * Air Velocity) + ({:.5f} * Steam Inlet Temp) + ({:.5f} * Steam Outlet Temp) + ({:.5f} * Steam Pressure) + ({:.5f} * Ambient Temp)".format(
    intercept, coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4]
)
print("Predicted Equation:\n", equation)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Generate random sample data for the variables
np.random.seed(42)  # For reproducibility

num_samples = 1000

air_velocity = np.random.uniform(low=0.5, high=2.5, size=num_samples)
steam_inlet_temp = np.random.uniform(low=100, high=200, size=num_samples)
steam_outlet_temp = np.random.uniform(low=200, high=300, size=num_samples)
steam_pressure = np.random.uniform(low=1, high=10, size=num_samples)
ambient_temp = np.random.uniform(low=20, high=30, size=num_samples)
moisture_content = np.random.uniform(low=0, high=1, size=num_samples)

# Combine the variables into a dataframe
data = pd.DataFrame({
    'Air Velocity': air_velocity,
    'Steam Inlet Temp': steam_inlet_temp,
    'Steam Outlet Temp': steam_outlet_temp,
    'Steam Pressure': steam_pressure,
    'Ambient Temp': ambient_temp,
    'Moisture Content': moisture_content
})

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Fit a linear regression model
X_train = train_data.drop('Moisture Content', axis=1)
y_train = train_data['Moisture Content']
X_train = sm.add_constant(X_train)  # Add a constant term for the intercept
model = sm.OLS(y_train, X_train)
results = model.fit()

# Print the summary report
print(results.summary())

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from PIL import Image


# Load the trained linear regression model coefficients and intercept
coefficients = np.array([coefficients[0], coefficients[1], coefficients[2], coefficients[3], coefficients[4]])  # Coefficients obtained from the trained model
intercept = regressor.intercept_  # Intercept obtained from the trained model

# Generate average value for steam pressure (just for demonstration purposes)
avg_steam_pressure = 5.5

# Create a Streamlit web app
st.title("Loss of Drying Prediction")

# Upload manufacturing image
image = Image.open(r"C:\Users\Harshita.Saxena\Downloads\manufacturing_img.jpeg")
st.image(image, use_column_width=True)

# Display subheading for external temperature
st.subheader("External Parameters")
ambient_temp = st.number_input("Ambient Temperature", value=25.0)


st.subheader("Controllable Parameters")
# User input for the controllable independent variables
air_velocity = st.number_input("Air Velocity", value=1.0)
steam_inlet_temp = st.number_input("Steam Inlet Temperature", value=150.0)
steam_outlet_temp = st.number_input("Steam Outlet Temperature", value=250.0)


# Display subheading for fixed input parameters
st.subheader("Fixed Input")

# Display the fixed value for steam pressure
st.write("Steam Pressure:", avg_steam_pressure)

# Create a dataframe with the user input and fixed values
input_data = pd.DataFrame({
    'Air Velocity': [air_velocity],
    'Steam Inlet Temp': [steam_inlet_temp],
    'Steam Outlet Temp': [steam_outlet_temp],
    'Steam Pressure': [avg_steam_pressure],
    'Ambient Temp': [ambient_temp]
})

# Add a constant term for the intercept
input_data['const'] = 1.0 #sm.add_constant(input_data)

# Predict the moisture content using the trained model
predicted_moisture_content = np.dot(input_data, np.append([intercept], coefficients))

# Display the predicted moisture content
st.subheader("Predicted Moisture Content")

if predicted_moisture_content > 10:
    st.markdown(
        f'<div style="background-color: red; padding: 10px; border-radius: 5px;">'
        f'<p style="color: white; font-weight: bold;">{predicted_moisture_content}</p>'
        f'<p style="color: white;">Outside Target Range</p>'
        f'</div>',
        unsafe_allow_html=True
    )
elif 5 <= predicted_moisture_content <= 10:
    st.markdown(
        f'<div style="background-color: yellow; padding: 10px; border-radius: 5px;">'
        f'<p style="font-weight: bold;">{predicted_moisture_content}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div style="background-color: green; padding: 10px; border-radius: 5px;">'
        f'<p style="font-weight: bold;">{predicted_moisture_content}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

# Button for issue input parameter recommendations
if predicted_moisture_content > 10:
    if st.button("Issue Input Parameter Recommendations"):
        st.write("Recommendation: Adjust the input parameters to bring the moisture content within the target range.")


# In[ ]:





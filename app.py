import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def get_model():
    return load_model("iris_model.keras")

model = get_model()

# Define the class names
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Set up the Streamlit app title and description
st.title("Iris Flower Species Prediction")
st.write("Enter the measurements of an Iris flower to predict its species.")

# Create input widgets for the four features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2)

# Create a button to make a prediction
if st.button('Predict'):
    # Prepare the input data for the model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make a prediction
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    
    # Display the result
    st.success(f"The predicted Iris species is: **{predicted_class_name}**")
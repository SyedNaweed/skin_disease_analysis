import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import joblib

# Load your skin disease detection model (.h5 file)
model = load_model('D:\Skindisease analysis\Skin_disease_stream\my_model.h5')  # Update the path accordingly

# Load LabelEncoder
label_encoder = joblib.load('D:\Skindisease analysis\Skin_disease_stream\label_encoder.pkl')  # Update with your label encoder file

# Define the function for skin disease prediction
def predict_skin_disease(image_path, model, label_encoder):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    processed_img = preprocess_input(np.array([img]))

    # Make predictions using the provided model
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_encoder.classes_[predicted_class_index]

    return predicted_class

# Streamlit app
def main():
    st.title("Skin Disease Analyzer")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Make a prediction when button is clicked
        if st.button('Analyze'):
            try:
                # Save the uploaded file locally
                temp_image_path = "temp_image.jpg"
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Use the prediction function
                predicted_result = predict_skin_disease(temp_image_path, model, label_encoder)

                # Display the predicted result
                st.success(f"Predicted Disease: {predicted_result}")

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == '__main__':
    main()

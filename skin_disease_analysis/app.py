from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

import joblib
import cv2
app = Flask(__name__)

# Load model architecture from JSON file
model = load_model('D:\Skindisease analysis\my_model.h5')



# Load LabelEncoder
le = joblib.load('D:\Skindisease analysis\label_encoder.pkl')  # Update with your label encoder file

# Define preprocess_image function
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

@app.route('/')
def index():
    return render_template('analyzer.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'photo' not in request.files:
        return jsonify({'result': 'No file part'})

    photo = request.files['photo']

    if photo.filename == '':
        return jsonify({'result': 'No selected file'})

    # Save the image temporarily (optional) or process it directly
    image_path = 'temp_image.jpg'  # Replace with your desired path or use temporary storage
    photo.save(image_path)

    # Use predict_skin_disease function for prediction
    predicted_result = predict_skin_disease(image_path, model, le)
    print(f"Predicted Result: {predicted_result}")  # Add print for debugging

    # Pass the result back as JSON to the front-end
    return jsonify({'result': predicted_result})

if __name__ == '__main__':
    app.run(debug=True)

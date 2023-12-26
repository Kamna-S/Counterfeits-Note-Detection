import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the trained model
model = load_model("my_model.h5")

img_height, img_width = 224, 224
# Define function for image prediction
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for the batch
    img_array /= 255.0  # Rescale to [0, 1]

    # Make prediction
    prediction = model.predict(img_array)

    # Return the predicted result
    return prediction[0, 0]


# Streamlit app
def main():
    # Title and file uploader
    st.title("Counterfeit Note Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image_path = "./temp_image.jpg"  # Save the uploaded image temporarily
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        # Process the image and make prediction
        prediction = predict_image(image_path)

        # Display the prediction result
        if prediction > 0.5:
            st.write("Predicted class: Real")
        else:
            st.write("Predicted class: Fake")


# Run the Streamlit app
if __name__ == "__main__":
    main()

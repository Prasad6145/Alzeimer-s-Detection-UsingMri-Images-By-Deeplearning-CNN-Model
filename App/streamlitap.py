# import streamlit as st
# import tensorflow as tf
# from keras.models import load_model
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the pre-trained model
# model = load_model(r"C:\Users\prasad k\Documents\Alzeimerdetection\alzheimers-detection\CNN\cnn_best_weights.hdf5")
# # model = load_model("model.h5")
# # Define the image size for model input
# IMG_SIZE = (128, 128)

# # Set the app title and sidebar
# # Add custom CSS for aesthetics
# st.markdown(
#     """
#     <style>
#     .title {
#         margin-top:0px;
#         color: #FF5733; /* Coral */
#         font-size: 40px;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 10px;
#     }
    
#     .text {
#         color: #EFA18A; /* Slate Gray */
#         font-size: 20px;
#         font-weight: italic;
#         text-align: center;
#         margin-bottom: 20px;
#     }
    
#     .uploaded-image {
#         width: 100%;
#         max-width: 500px;
#         margin-bottom: 20px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#     }
    
#     .prediction {
#         color: #FF5733; /* Coral */
#         font-size: 24px;
#         font-weight: bold;
#         margin-bottom: 10px;
#         text-align: center;
#     }
    
#     .confidence {
#         color: #FF5600; /* Coral */
#         font-size: 18px;
#         margin-bottom: 20px;
#         text-align: center;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # st.set_option('deprecation.showPyplotGlobalUse', False)


# # Display the title
# st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
# st.markdown("<h1 class='text'>Alzheimer's Disease Prediction is a web application that utilizes a pre-trained deep learning model to predict the presence of Alzheimer's disease based on uploaded brain ultrasound images. Users can upload an image through the sidebar and the app will process the image using the trained model.</h1>", unsafe_allow_html=True)

# st.sidebar.title("Upload Image")
# st.sidebar.markdown("Please upload an image.")


# def preprocess_image(image):
#     # plt.imsave('image2.jpg', image)
#     img_array = np.array(image)
#     rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
#     img = Image.fromarray(img_array.astype('uint8'))


#     # img.save('output1.jpg')  # Save the image to a file

#     img_array = np.expand_dims(rgb_image, axis=0)
#     return img_array




 
# def predict(image):
#     img_array = preprocess_image(image)
#     prediction = model.predict(img_array)
#     # print(prediction)
#     predicted_idx = np.argmax(prediction, axis=1)[0]
#     return predicted_idx

# # Display the file uploader
# uploaded_file = st.sidebar.file_uploader(label="", type=['jpg', 'jpeg', 'png'])

# # Make predictions and display the result
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     predicted_idx = predict(image)
    
#     class_labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
#     predicted_label = class_labels[predicted_idx]

#     st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)

# else:
#     st.sidebar.write("Please upload an image.")
# import streamlit as st
# import tensorflow as tf
# from keras.models import load_model
# from PIL import Image
# import numpy as np

# # Load the pre-trained model
# model = load_model(r"C:\Users\prasad k\Documents\Alzeimerdetection\alzheimers-detection\App\model.h5")

# # Define the image size for model input
# IMG_SIZE = (128, 128)

# # Set the app title and sidebar
# st.markdown(
#     """
#     <style>
#     .title {
#         margin-top:0px;
#         color: #FF5733; /* Coral */
#         font-size: 40px;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 10px;
#     }
    
#     .text {
#         color: #EFA18A; /* Slate Gray */
#         font-size: 20px;
#         font-weight: italic;
#         text-align: center;
#         margin-bottom: 20px;
#     }
    
#     .uploaded-image {
#         width: 100%;
#         max-width: 500px;
#         margin-bottom: 20px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#     }
    
#     .prediction {
#         color: #FF5733; /* Coral */
#         font-size: 24px;
#         font-weight: bold;
#         margin-bottom: 10px;
#         text-align: center;
#     }
    
#     .confidence {
#         color: #FF5600; /* Coral */
#         font-size: 18px;
#         margin-bottom: 20px;
#         text-align: center;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Display the title
# st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
# st.markdown("<h1 class='text'>This web application uses a pre-trained deep learning model to predict Alzheimer's disease based on uploaded brain ultrasound images. Please upload an image to get a prediction.</h1>", unsafe_allow_html=True)

# st.sidebar.title("Upload Image")
# st.sidebar.markdown("Please upload an image to analyze.")

# from PIL import Image
# import numpy as np

# def preprocess_image(image):
#     # Convert PIL Image to RGB
#     image = image.convert('RGB')
    
#     # Resize the image to the input size your model expects
#     image = image.resize((128, 128))
    
#     # Convert image to a NumPy array and normalize it
#     img_array = np.array(image) / 255.0

#     # Add batch dimension => shape becomes (1, 128, 128, 3)
#     img_array = np.expand_dims(img_array, axis=0)

#     return img_array

# def predict(image):
#     img_array = preprocess_image(image)
#     prediction = model.predict(img_array)
#     predicted_idx = np.argmax(prediction, axis=1)[0]
#     return predicted_idx



# uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     predicted_idx = predict(image)

#     # Display the uploaded image
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     # Make the prediction
#     predicted_idx = predict(image)
    
#     # Class labels for prediction
#     class_labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
#     predicted_label = class_labels[predicted_idx]

#     # Show prediction result
#     st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)

# else:
#     # Display message if no image is uploaded
#     st.sidebar.write("Please upload an image to make a prediction.")
#     st.write("Awaiting image upload...")





import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model(r"C:\Users\prasad k\Documents\Alzeimerdetection\alzheimers-detection\CNN\cnn_best_weights.hdf5")

# Define the image size for model input
IMG_SIZE = (128, 128)

# Custom CSS styling
st.markdown(
    """
    <style>
    .title {
        margin-top:0px;
        color: #FF5733;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .text {
        color: #EFA18A;
        font-size: 20px;
        font-weight: italic;
        text-align: center;
        margin-bottom: 20px;
    }
    .uploaded-image {
        width: 100%;
        max-width: 500px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .prediction {
        color: #FF5733;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    .details-box {
        background-color: #fff3e6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ffc9a3;
        font-size: 16px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='text'>Upload a Brain MRI Image Check Statges Of Alzheimer Disease Using AI.</h1>", unsafe_allow_html=True)

# Sidebar upload
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader(label="Choose a brain scan image", type=['jpg', 'jpeg', 'png'])

# Image preprocessing
def preprocess_image(image):
    img_array = np.array(image.resize(IMG_SIZE).convert("L"))  # convert to grayscale
    rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
    img_array = np.expand_dims(rgb_image, axis=0)
    return img_array

# Prediction
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    return predicted_idx

# Class labels and descriptions
class_labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
class_descriptions = {
    'No Impairment': """
    <div class='details-box'>
    <h3 style="color:#333;">No Impairment</h3>
    <ul style="font-size:17px; color:#222;">
        <li><b>Cognitive Status:</b> Normal</li>
        <li><b>Symptoms:</b> No noticeable memory or thinking problems</li>
        <li><b>Recommendation:</b> Maintain a healthy lifestyle and routine checkups</li>
    </ul>
    </div>
    """,
    'Very Mild Impairment': """
    <div class='details-box'>
    <h3 style="color:#333;">Very Mild Impairment</h3>
    <ul style="font-size:17px; color:#222;">
        <li><b>Cognitive Status:</b> Slight decline</li>
        <li><b>Symptoms:</b> Minor forgetfulness or difficulty recalling names</li>
        <li><b>Recommendation:</b> Regular monitoring and mental stimulation activities</li>
    </ul>
    </div>
    """,
    'Mild Impairment': """
    <div class='details-box'>
    <h3 style="color:#333;">Mild Impairment</h3>
    <ul style="font-size:17px; color:#222;">
        <li><b>Cognitive Status:</b> Noticeable memory loss</li>
        <li><b>Symptoms:</b> Trouble with familiar tasks, confusion with time/place</li>
        <li><b>Recommendation:</b> Medical diagnosis recommended. Cognitive exercises can help</li>
    </ul>
    </div>
    """,
    'Moderate Impairment': """
    <div class='details-box'>
    <h3 style="color:#333;">Moderate Impairment</h3>
    <ul style="font-size:17px; color:#222;">
        <li><b>Cognitive Status:</b> Middle stage Alzheimer's</li>
        <li><b>Symptoms:</b> Memory loss worsens, difficulty recognizing people, speech problems</li>
        <li><b>Recommendation:</b> Professional care needed. Structured routines and support essential</li>
    </ul>
    </div>
    """
}

# Main App Logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    predicted_idx = predict(image)
    predicted_label = class_labels[predicted_idx]

    # Show prediction
    st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)

    # Show detailed info
    st.markdown(class_descriptions[predicted_label], unsafe_allow_html=True)

else:
    st.sidebar.write("Upload an image to get started.")

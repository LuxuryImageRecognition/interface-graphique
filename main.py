import base64
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from prediction import get_prediction


def pil_base64(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


st.markdown(
    """
<style>

 .reportview-container {
        background: #fffffc;
        }
.reportview-container .markdown-text-container {
}


.sidebar .sidebar-content {
    background-image: linear-gradient(#E6E2DD,#E6E2DD);
   
    color: black;
}




</style>
""",
    unsafe_allow_html=True,
)

image = Image.open('image6.png')
show = st.image(image, use_column_width=True)
u_img = None

st.sidebar.title("Upload Image")

# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    image = np.asarray(u_img) / 255

st.sidebar.text('\n')
if st.sidebar.button("Click Here to Classify"):

    if uploaded_file is None:
        st.sidebar.header("Please upload an Image to Classify")

    else:
        with st.spinner('Classifying...'):
            encoded_img = pil_base64(u_img)
            u_img.save("temp.jpg")

            with open("temp.jpg", 'rb') as ff:
                encoded_image = ff.read()
            prediction = get_prediction(encoded_image)
            predicted_name = prediction.payload[0].display_name
            prediction_score = prediction.payload[0].classification
            print(predicted_name)
            print(prediction_score)
            # mettre la prediction
            prediction = 0.7
            st.success('Done!')

        st.sidebar.header("Algorithm Predicts: ")
        probability = "{:.3f}".format(float(prediction * 100))

        # Classify the bag being present in the picture if prediction > 0.5
        if prediction > 0.5:
            st.sidebar.markdown("It's a '?' picture.")
            st.sidebar.markdown('**Probability in %: **')
            st.sidebar.text(probability)
        else:
            st.sidebar.markdown(" There is no bag in this picture ")
            st.sidebar.markdown('**Probability in %: **')
            st.sidebar.text(probability)

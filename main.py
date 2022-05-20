import streamlit as st
from PIL import Image
import numpy as np

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

st.sidebar.title("Upload Image")

#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )
if uploaded_file is not None:

    u_img = Image.open(uploaded_file)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    image = np.asarray(u_img)/255

st.sidebar.text('\n')
if st.sidebar.button("Click Here to Classify"):

    if uploaded_file is None:
        st.sidebar.header("Please upload an Image to Classify")

    else:
        with st.spinner('Classifying...'):
            #mettre la prediction
            prediction = 0.7
            st.success('Done!')

        st.sidebar.header("Algorithm Predicts: ")
        probability = "{:.3f}".format(float(prediction*100))

# Classify the bag being present in the picture if prediction > 0.5

        if prediction > 0.5:

            st.sidebar.markdown("It's a '?' picture.")
            st.sidebar.markdown('**Probability in %: **')
            st.sidebar.text(probability)


        else:

            st.sidebar.markdown(" There is no bag in this picture ")
            st.sidebar.markdown('**Probability in %: **')
            st.sidebar.text(probability)

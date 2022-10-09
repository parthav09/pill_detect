import numpy as np
import streamlit as st
import cv2

uploaded_file = st.file_uploader("Choose a image file", type="bmp")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    # st.image(gray)
    (T, threshinv) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    xaxis = cv2.Sobel(threshinv, ddepth=cv2.CV_64F, dx=1, dy=0)  # identifying horizontal changes
    yaxis = cv2.Sobel(threshinv, ddepth=cv2.CV_64F, dx=0, dy=1)  # identifying vertical changes
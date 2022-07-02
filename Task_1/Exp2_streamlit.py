# load image as gray scale
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import cv2

img = cv2.imread("pout.jpg", 0)

# plot histogram of image
fig = plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.hist(img.ravel(), 256, [0, 256], label="Before_histogram")
plt.title("Before")

# get min and max value of image
a = img.min()
b = img.max()

# input new value for min and max
c = st.slider("Please enter new min", min_value=0, max_value=255, step=1, value=0)
d = st.slider("Please enter new max", min_value=0, max_value=255, step=1, value=255)
x = (d - c) / (b - a)
# map the image pixels to the new range
img_eq = (x * (img - a)) + c

img_eq = img_eq.astype(np.uint8)

st.image([img, img_eq], width=350)
st.write("a: ", a, "b: ", b, "c: ", c, "d: ", d)

plt.subplot(122)

plt.hist(img_eq.ravel(), 256, [0, 256], label="After_histogram")
plt.title("After")
st.pyplot(fig)

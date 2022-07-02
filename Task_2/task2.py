#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st

# using a gray scale image
img = cv2.imread("images.jpeg", 0)
# st.image(img, caption="Original Image", width=300)
st.write("# Experiment 1:")
st.write("### a. Creating noise and adding it to the image")
# add salt & pepper noise to the given image
temp_img = img.copy()
salt_pepper_noise = cv2.randn(temp_img, 0, 255)
st.image(salt_pepper_noise, caption="Salt & Pepper Noise", width=300)
noisy_img = img + salt_pepper_noise
st.image([img, noisy_img], caption=["Original Image", "Noisy Image"], width=300)

#%%
# sliders
# to change the average Filter window size
avg_window_size = st.sidebar.slider(
    "Average Filter Window Size", min_value=1, max_value=15, value=5, step=2
)
# to change the median Filter window size
med_window_size = st.sidebar.slider(
    "Median Filter Window Size", min_value=1, max_value=15, value=5, step=2
)
# to change the rank order Filter window size
rank_window_size = st.sidebar.slider(
    "Rank Order Filter Window Size", min_value=1, max_value=15, value=5, step=2
)
# to change the rank order Filter rank
rank_order_rank = st.sidebar.slider(
    "Rank Order Filter Rank",
    min_value=0,
    max_value=(rank_window_size**2) - 1,
    value=(rank_window_size * rank_window_size) // 2,
    step=1,
)
# to change the laplacian Filter window size
lap_window_size = st.sidebar.slider(
    "Laplacian Filter Window Size", min_value=1, max_value=15, value=5, step=2
)
# to change the gaussian Filter window size
gauss_window_size = st.sidebar.slider(
    "Gaussian Filter Window Size", min_value=1, max_value=15, value=5, step=2
)
# to change the lap_window_size
lap_of_gaussian_window_size = st.sidebar.slider(
    "Laplacian of Gaussian Filter Window Size",
    min_value=1,
    max_value=15,
    value=5,
    step=2,
)

#%%
# applying different smoothing filters with different window size
# average filter
st.write("### b. Applying Average Filter")
average_filter = cv2.blur(noisy_img, (avg_window_size, avg_window_size))
st.image(
    [img, average_filter],
    caption=["Original Image", f"Average Filter_{avg_window_size}"],
    width=300,
)
#%%
# median filter
st.write("### c. Applying Median Filter")
median_filter = cv2.medianBlur(noisy_img, med_window_size)
st.image(
    [img, median_filter],
    caption=["Original Image", f"Median Filter_{med_window_size}"],
    width=300,
)
#%%
# rank order filter
from PIL import Image, ImageFilter

st.write("### d. Applying Rank Order Filter")

# convert the noisy image to PIL format
im = Image.fromarray(np.uint8(noisy_img))
rank_order_filter = im.filter(
    ImageFilter.RankFilter(size=rank_window_size, rank=rank_order_rank)
)
st.image(
    [img, rank_order_filter],
    caption=[
        "Original Image",
        f"Rank Order Filter_{rank_window_size}_{rank_order_rank}",
    ],
    width=300,
)
#%%
#  Explain briefly the following questions:
# Q1. Which smoothing filter removes the noise in a better way? Why?
# A1. median filter removes the noise in a better way than average filter.because median filter is a non-linear filter
# and isn't affected by noise.
# Q2. What is the effect of changing the window size?
# A2. changing the window size of the filter makes the image more blurry (smoother).
# Q3. When utilizing the rank order filter, what is the effect of changing the rank?
# A3. as the rank increases, the image becomes more blurry. however, too high a rank can cause the image to become burned.
# and too low a rank can cause the image to become dark (black).
st.write("## Explain briefly the following questions:")
st.write("---------")
st.write("### Q1. Which smoothing filter removes the noise in a better way? Why?")
st.write(
    "##### A. **Median** filter removes the noise in a better way than average filter. because it is a non-linear filter and isn't affected by noise."
)
st.write("### Q2. What is the effect of changing the window size?")
st.write(
    "##### A. Changing the window size of the filter makes the image more **blurry** (smoother)."
)
st.write(
    "### Q3. When utilizing the rank order filter, what is the effect of changing the rank?"
)
st.write(
    "##### A. choosing higher rank can cause the image to become burned. and choosing low rank can cause the image to become dark (black)."
)
#%%
# Apply different sharpening filters (Perwitt, Sobel, laplacian, log).
st.write("# Experiment 2:")
st.write("### a. Applying Perwitt Filter")
# applying perwitt filter
perwitt_filter_y = cv2.filter2D(
    src=img, ddepth=-1, kernel=np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
)
perwitt_filter_x = cv2.filter2D(
    src=img, ddepth=-1, kernel=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
)
perwitt_filter = perwitt_filter_x + perwitt_filter_y
st.image(
    [img, perwitt_filter, perwitt_filter_x, perwitt_filter_y],
    caption=[
        "Original Image",
        f"Perwitt Filter",
        f"Perwitt Filter_x",
        f"Perwitt Filter_y",
    ],
    width=300,
)
#%%
# applying sobel filter
st.write("### b. Applying Sobel Filter")
sobel_filter_y = cv2.filter2D(
    src=img, ddepth=-1, kernel=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
)
sobel_filter_x = cv2.filter2D(
    src=img, ddepth=-1, kernel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
)
sobel_filter = sobel_filter_x + sobel_filter_y
st.image(
    [img, sobel_filter, sobel_filter_x, sobel_filter_y],
    caption=[
        "Original Image",
        f"Sobel Filter",
        f"Sobel Filter_x",
        f"Sobel Filter_y",
    ],
    width=300,
)
#%%
# applying laplacian filter
st.write("### c. Applying Laplacian Filter")
laplacian_filter = cv2.Laplacian(img, cv2.CV_8U, ksize=lap_window_size)
st.image(
    [img, laplacian_filter],
    caption=["Original Image", f"Laplacian Filter"],
    width=300,
)
#%%
# applying laplacian of gaussian filter
st.write("### d. Applying Laplacian of Gaussian Filter")
gaussian_filter = cv2.GaussianBlur(img, (gauss_window_size, gauss_window_size), 0)
lap_of_gaussian_filter = cv2.Laplacian(
    gaussian_filter, cv2.CV_8U, ksize=lap_of_gaussian_window_size
)
st.image(
    [img, gaussian_filter, lap_of_gaussian_filter],
    caption=["Original Image", f"gaussian_filter", "lap_of_gaussian_filter"],
    width=300,
)
#%%

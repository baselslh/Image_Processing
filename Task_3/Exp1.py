"""
1. Read an image that contains lines of several lengths.
2. Convert into greyscale image.
3. Find all the edge points in the image using any suitable edge detection scheme.
4. Apply Hough transform to detect the lines in the image.
5. After calculating the Hough space, the next step is to find the peaks by using a threshold T that represent the minimum number of intersections to detect a line.
6. Try different values of T and for each value display the original image and highlight the detected lines on it.
"""
#%%
import cv2
import numpy as np
import streamlit as st

# Read an image and convert it to greyscale
img = cv2.imread("sudoku.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#%%
# create a select box to choose the edge detection method
edge_detection_method = st.selectbox(
    "Select the edge detection method",
    ("Canny", "Sobel", "Laplacian", "Scharr", "Prewitt", "Roberts"),
)
# create a function to apply the selected edge detection method
def edge_detection(method):
    if method == "Canny":
        edges, default_hough = canny_edge_detection()
    elif method == "Sobel":
        edges, default_hough = sobel_edge_detection()
    elif method == "Laplacian":
        edges, default_hough = laplacian_edge_detection()
    elif method == "Scharr":
        edges, default_hough = scharr_edge_detection()
    elif method == "Prewitt":
        edges, default_hough = prewitt_edge_detection()
    elif method == "Roberts":
        edges, default_hough = roberts_edge_detection()
    return edges, default_hough


#%%
# create a function to apply canny edge detection
def canny_edge_detection():
    # print how does the canny edge detection works
    st.markdown(
        """
        Canny edge detection is a multi-stage algorithm.
        1. First a Gaussian filter is applied to the image to remove noise.
        2. Then a Sobel operator is applied to find the gradients.(dx, dy)
        3. Then a non-maximum suppression is applied (The result you get is a binary image with "thin edges").
        4. Then a double threshold is applied to the image to get the final result.
              
                IF the gradient is greater than the high threshold,
                 Then it is accepted as an edge.
                ELSE IF the gradient is less than the high threshold 
                 BUT greater than the low threshold, AND is connected to an edge,
                    Then it is accepted as an edge
                ELSE it is rejected as an edge
        """
    )
    # create a slider to choose the lower threshold
    low_threshold = st.sidebar.slider(
        "Low Threshold", min_value=0, max_value=255, value=50
    )
    # create a slider to choose the upper threshold
    high_threshold = st.sidebar.slider(
        "High Threshold", min_value=0, max_value=255, value=150
    )
    # Find all the edge points in the image using canny.
    edges = cv2.Canny(image=gray, threshold1=low_threshold, threshold2=high_threshold)
    st.image(
        [img, edges],
        caption=["Original Image", f"Canny Edge Detection"],
        width=300,
    )
    default_hough = 169
    return edges, default_hough


#%%
# create a function to apply sobel edge detection
def sobel_edge_detection():
    # create a slider to choose the kernel size
    kernel_size = st.sidebar.slider(
        "Kernel Size", min_value=1, max_value=7, value=3, step=2
    )

    # Find all the edge points in the image using sobel.
    sobelx = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=kernel_size)
    sobely = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=kernel_size)

    # normalize the image
    sobelx = cv2.normalize(sobelx, sobelx, 0, 255, cv2.NORM_MINMAX)
    sobely = cv2.normalize(sobely, sobely, 0, 255, cv2.NORM_MINMAX)

    # Combine the two images
    edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.97, 0)

    st.image(
        [img, edges],
        caption=["Original Image", f"Sobel Edge Detection"],
        width=300,
    )
    # Apply threshold to the image to get the final output.
    # create a slider to choose the threshold
    threshold = st.sidebar.slider("Threshold", min_value=0, max_value=254, value=60)

    _, dst = cv2.threshold(
        src=edges,
        thresh=threshold,
        maxval=max(edges.max(), 255),
        type=cv2.THRESH_BINARY,
    )
    st.image(
        [img, dst],
        caption=["Original Image", f"Sobel Edge Detection_threshold"],
        width=300,
    )
    default_hough = 260

    return dst, default_hough


#%%
# create a function to apply laplacian edge detection
def laplacian_edge_detection():
    # create a slider to choose the kernel size
    kernel_size = st.sidebar.slider(
        "Kernel Size", min_value=1, max_value=15, value=5, step=2
    )
    # Find all the edge points in the image using laplacian.
    edges = cv2.Laplacian(src=gray, ddepth=cv2.CV_8U, ksize=kernel_size)
    # st.write(edges)
    st.image(
        [img, edges],
        caption=["Original Image", f"Laplacian Edge Detection"],
        width=300,
    )
    # Apply threshold to the image to get the final output.
    # create a slider to choose the threshold
    threshold = st.sidebar.slider("Threshold", min_value=0, max_value=254, value=120)

    _, dst = cv2.threshold(
        src=edges,
        thresh=threshold,
        maxval=max(edges.max(), 255),
        type=cv2.THRESH_BINARY,
    )
    st.image(
        [img, dst],
        caption=["Original Image", f"Laplacian Edge Detection_threshold"],
        width=300,
    )

    default_hough = 363
    return dst, default_hough


#%%
# create a function to apply scharr edge detection
def scharr_edge_detection():
    # create a slider to choose the scale
    scale = st.sidebar.slider("Scale", min_value=1, max_value=3, value=1, step=1)

    # Find all the edge points in the image using scharr.
    sharrx = cv2.Scharr(src=gray, ddepth=cv2.CV_8U, dx=1, dy=0, scale=scale)
    sharry = cv2.Scharr(src=gray, ddepth=cv2.CV_8U, dx=0, dy=1, scale=scale)

    # normalize the image
    sharrx = cv2.normalize(sharrx, sharrx, 0, 255, cv2.NORM_MINMAX)
    sharry = cv2.normalize(sharry, sharry, 0, 255, cv2.NORM_MINMAX)

    # Combine the two images
    edges = cv2.addWeighted(sharrx, 0.5, sharry, 0.5, 0)

    st.image(
        [img, edges],
        caption=["Original Image", f"Scharr Edge Detection"],
        width=300,
    )
    # Apply threshold to the image to get the final output.
    # create a slider to choose the threshold
    threshold = st.sidebar.slider("Threshold", min_value=0, max_value=254, value=100)

    _, dst = cv2.threshold(
        src=edges,
        thresh=threshold,
        maxval=max(edges.max(), 255),
        type=cv2.THRESH_BINARY,
    )
    st.image(
        [img, dst],
        caption=["Original Image", f"Scharr Edge Detection_threshold"],
        width=300,
    )

    default_hough = 369
    return dst, default_hough


#%%
# create a function to apply prewitt edge detection
def prewitt_edge_detection():

    # Find all the edge points in the image using prewitt.
    prewittx = cv2.filter2D(
        src=gray,
        ddepth=cv2.CV_8U,
        kernel=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    )
    prewitty = cv2.filter2D(
        src=gray,
        ddepth=cv2.CV_8U,
        kernel=np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    )

    # normalize the image
    prewittx = cv2.normalize(prewittx, prewittx, 0, 255, cv2.NORM_MINMAX)
    prewitty = cv2.normalize(prewitty, prewitty, 0, 255, cv2.NORM_MINMAX)

    # Combine the two images
    edges = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)

    st.image(
        [img, edges],
        caption=["Original Image", f"Prewitt Edge Detection"],
        width=300,
    )
    # Apply threshold to the image to get the final output.
    # create a slider to choose the threshold
    threshold = st.sidebar.slider("Threshold", min_value=0, max_value=254, value=27)

    _, dst = cv2.threshold(
        src=edges,
        thresh=threshold,
        maxval=max(edges.max(), 255),
        type=cv2.THRESH_BINARY,
    )
    st.image(
        [img, dst],
        caption=["Original Image", f"Prewitt Edge Detection_threshold"],
        width=300,
    )

    default_hough = 262

    return dst, default_hough


#%%
# create a function to apply roberts edge detection
def roberts_edge_detection():

    # Find all the edge points in the image using roberts.
    robertsx = cv2.filter2D(
        src=gray,
        ddepth=cv2.CV_8U,
        kernel=np.array([[1, 0], [0, -1]]),
    )
    robertsy = cv2.filter2D(
        src=gray,
        ddepth=cv2.CV_8U,
        kernel=np.array([[0, 1], [-1, 0]]),
    )

    # normalize the image
    robertsx = cv2.normalize(robertsx, robertsx, 0, 255, cv2.NORM_MINMAX)
    robertsy = cv2.normalize(robertsy, robertsy, 0, 255, cv2.NORM_MINMAX)

    # Combine the two images
    edges = cv2.addWeighted(robertsx, 0.5, robertsy, 0.5, 0)

    st.image(
        [img, edges],
        caption=["Original Image", f"Roberts Edge Detection"],
        width=300,
    )
    # Apply threshold to the image to get the final output.
    # create a slider to choose the threshold
    threshold = st.sidebar.slider("Threshold", min_value=0, max_value=254, value=5)

    _, dst = cv2.threshold(
        src=edges,
        thresh=threshold,
        maxval=max(edges.max(), 255),
        type=cv2.THRESH_BINARY,
    )
    st.image(
        [img, dst],
        caption=["Original Image", f"Roberts Edge Detection_threshold"],
        width=300,
    )

    default_hough = 381

    return dst, default_hough


#%%
# get the edges of the image
edges, default_hough = edge_detection(edge_detection_method)

# create a slider to choose the hough lines threshold
hough_threshold = st.sidebar.slider(
    "Hough Threshold", min_value=1, max_value=1000, value=default_hough, step=1
)
# apply hough transform to detect the lines in the image.
lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=hough_threshold)
# After calculating the Hough space, the next step is to find the peaks by using a threshold T that represent the minimum number of intersections to detect a line.
# Try different values of T and for each value display the original image and highlight the detected lines on it.
if lines is None:
    st.write("Please reduce the threshold value")
else:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # update the image
        cv2.line(img, (x1, y1), (x2, y2), (75, 75, 255), 2)

    # convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HoughLines_image = cv2.imshow("Hough Lines", img)
    st.image(
        img,
        caption=["Hough Transform"],
        width=300,
    )

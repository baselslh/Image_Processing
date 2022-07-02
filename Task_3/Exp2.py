"""
1. Read an image and convert it to gray scale.
2. Let the user to interactively select set of points or locations inside the image (seed points).
3. Also, let the user to decide a threshold value T.
4. Apply region growing algorithm to find the region of each seed point.
5. Display the original image and highlight the detected regions on it.
"""
#%%
import cv2
import numpy as np
import streamlit as st

# Read an image and convert it to gray scale.
img = cv2.imread("1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#%%
# use plotly to display the image
import plotly.express as px

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = px.imshow(gray, binary_string=True)
# dont show x and y coordinates in the plotly graph
fig.update_layout(
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    height=750,
    width=750,
    title={
        "text": "Gray scale image",
        "y": 0.98,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": dict(size=30, color="rgb(250, 115, 115)"),
    },
)
st.plotly_chart(fig)
#%%
# Let the user to interactively select set of points or locations inside the image (seed points).
# inital seed points
INP_SEEDS = [(176, 255), (229, 405), (347, 165)]


# create a function to create 2 input boxes for for x and y coordinates of the seed points
def create_input_boxes(num_seed_points=1):
    x_coords = []
    y_coords = []
    for i in range(num_seed_points):
        a = INP_SEEDS[i][0] if i < len(INP_SEEDS) else 0
        b = INP_SEEDS[i][1] if i < len(INP_SEEDS) else 0
        x_coords.append(
            st.sidebar.number_input(
                f"X-Coordinate of Seed Point {i+1}",
                value=a,
                max_value=img.shape[1],
                min_value=0,
            )
        )
        y_coords.append(
            st.sidebar.number_input(
                f"Y-Coordinate of Seed Point {i+1}",
                value=b,
                max_value=img.shape[0],
                min_value=0,
            )
        )
    return x_coords, y_coords


# create a slider to select the number of seed points
num_seed_points = st.sidebar.slider(
    "Number of Seed Points", min_value=1, max_value=10, value=1, step=1
)
# get x and y coordinates of the seed points
x_coords, y_coords = create_input_boxes(num_seed_points)
# %%
# create a function to find the region of each seed point
def region_growing(gray, x_coords, y_coords, threshold):
    # create a copy of the image
    img_copy = gray.copy()

    no_of_splits = len(x_coords)

    color = 255
    visited = np.zeros((gray.shape[0], gray.shape[1]))
    # loop over the seed points
    regions = [[x, y] for y, x in zip(x_coords, y_coords)]
    for x, y in regions:
        x = int(x)
        y = int(y)

        # get the value of the seed point
        center = gray[x, y]
        # create a map of the region
        RMAP = np.zeros((gray.shape[0], gray.shape[1]))
        # print(RMAP)
        RMAP[x, y] = 1
        # grow the region
        neighbors = find_neighbors(img_copy, RMAP, x, y, threshold, visited, center)
        # get the neighbors of all the negihbors
        for neighbor in neighbors:
            x1, y1 = neighbor
            new = find_neighbors(img_copy, RMAP, x1, y1, threshold, visited, center)
            _ = [neighbors.append(neighbor) for neighbor in new]

        # find the region of the seed point
        img_copy[RMAP == 1] = color
        color -= 255 / no_of_splits
    return img_copy


# create a function to find the neighbors of each seed point that are not yet visited and are within the threshold
def find_neighbors(image, RMAP, x, y, threshold, visited, center):
    # get size of image
    a = image.shape[0]
    b = image.shape[1]

    # cast values to int
    x = int(x)
    y = int(y)

    # list of neighbors
    neighbors = []

    # grow the region
    for i in range(-1, 2):
        for j in range(-1, 2):
            # if the neighbor is within the image and not visited
            if (
                (x + i) > 0
                and (y + j) > 0
                and (x + i) < a
                and (y + j) < b
                and not visited[x + i, y + j]
            ):
                diff = abs(int(image[x + i, y + j]) - int(center))
                if diff < threshold:
                    neighbors.append([x + i, y + j])
                    visited[x + i, y + j] = 1
                    RMAP[x + i, y + j] = 1

    return neighbors


#%%
# Also, let the user to decide a threshold value T.
threshold = st.sidebar.slider(
    "Threshold Value", min_value=0, max_value=255, value=100, step=1
)

# Display the original image and highlight the detected regions on it.
img = region_growing(gray, x_coords, y_coords, threshold)

#%%
# create a plotly graph
fig = px.imshow(img, binary_string=True)
# dont show x and y coordinates in the plotly graph
fig.update_layout(
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    height=750,
    width=750,
    title={
        "text": "After Region Growing",
        "y": 0.98,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": dict(size=30, color="rgb(250, 115, 115)"),
    },
)
st.plotly_chart(fig)
#%%

# Image_Processing
 
## General info 
This are the assignments that i had to complete in image processing 1 course

## 1. Task 1
a. Experiment 1:

- Choose an image with low contrast and work with it as gray image
- Plot its histogram
- Plot its cumulative histogram
- Perform histogram equalization
- Plot the modified image
- Plot the modified image histogram
- Plot the modified image cumulative histogram

b. Experiment 2:
- Choose an image with low contrast whose pixels not occupy the full dynamic range and work with it as gray image.
- Deduce the original image range from a to b [min and max of image pixels]
- Plot its histogram
- Apply contrast stretching on image pixels to utilize the full dynamic range [0,255]
- Display the new image
- Plot the modified image histogram

## 2. Task 2
a. Experiment 1 (smoothing):
- Using any gray scale image:
- Add salt & pepper noise to the given image.
- for the noisy image Apply different smoothing filters with different window size:
- (Average, median, rank order with different ranks).
- Explain briefly the following questions:
  1. Which smoothing filter removes the noise in a better way? Why?
  2. What is the effect of changing the window size?
  3. When utilizing the rank order filter, what is the effect of changing the rank?

b. Experiment 2 (sharpening):
- Using any gray scale image:
- Apply different sharpening filters (Perwitt, Sobel, laplacian, log).

## 3. Task 3
Experiment 1:

- Read an image that contains lines of several lengths.
- Convert into greyscale image.
- Find all the edge points in the image using any suitable edge detection scheme.
- Apply Hough transform to detect the lines in the image.
- After calculating the Hough space, the next step is to find the peaks by using a threshold T that represent the minimum number of intersections to detect a line.
- Try different values of T and for each value display the original image and highlight the detected lines on it.
 

Experiment 2:

- Read an image and convert it to gray scale.
- Let the user to interactively select set of points or locations inside the image (seed points).
- Also, let the user to decide a threshold value T.
- Apply region growing algorithm to find the region of each seed point.
- Display the original image and highlight the detected regions on it.

## 4. Task 4
- Use any CNN trained for MNIST handwritten digit classification (that classify between the digits from 0 to 9).
- After training the model, evaluate it using the testing data and get the predicted class of the testing image samples.
- By comparing the predicted class labels with the actual class labels (Ground Truth), build the Confusion Matrix of the ten classes (digits from 0 to 9).
- For each class (digit):
  - Calculate the precision, recall, F1 score.


## 5. Project
- Facial Expression Recognition (FER)
 
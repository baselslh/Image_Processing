import cv2
import numpy as np
from keras.preprocessing import image
import warnings

warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# load model
model = load_model("model_optimal.h5")


face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])
label_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}
while True:
    (
        ret,
        test_img,
    ) = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (72, 50, 72), thickness=5)
        roi_gray = gray_img[
            y : y + w, x : x + h
        ]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        # img_pixels = img_pixels.reshape(1, 48, 48, 1)

        predictions = model.predict(img_pixels)
        predictions = list(predictions[0])
        # find max indexed array
        # max_index = np.argmax(predictions[0])

        img_index = predictions.index(max(predictions))

        cv2.putText(
            test_img,
            label_dict[img_index],
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (191, 64, 191),
            2,
        )

    # resized_img = cv2.resize(test_img, (1000, 700))
    FRAME_WINDOW.image(test_img, channels="BGR", width=750)

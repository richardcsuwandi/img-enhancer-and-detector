import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np

# Import the Haar Cascade Classifier model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def detect_faces(my_image):
    new_img = np.array(my_image.convert("RGB"))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    count = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
    return img, count

def detect_eyes(my_image):
    new_img = np.array(my_image.convert("RGB"))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    count = 0
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
        count += 1
    return img, count

def cartonize_image(my_image):
    new_img = np.array(my_image.convert("RGB"))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def cannize_image(my_image):
    new_img = np.array(my_image.convert("RGB"))
    img = cv2.cvtColor(new_img,1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

def main():
    st.title("Image Enhancer and Detector 👀")
    st.markdown("By Richard Cornelius Suwandi")
    st.sidebar.title("Image Enhancer and Detector 👀")
    st.sidebar.markdown("By Richard Cornelius Suwandi")

    image_file = st.sidebar.file_uploader("Upload image", type=["jpg","png","jpeg"])

    task = ["Image Enhancement", "Image Detection"]
    choice = st.sidebar.selectbox("Choose task", task)

    if image_file is not None:
        my_image = Image.open(image_file)
        st.subheader("Original")
        st.image(my_image, width=500)
        if choice == "Image Enhancement":
            st.subheader("Result")
            types = ["Gray-Scale", "Contrast", "Brightness", "Color Balance", "Blur", "Cartoonize"]
            enhance_type = st.sidebar.radio("Enhancement Type", types)
            # Gray-scale
            if enhance_type == "Gray-Scale":
                new_img = np.array(my_image.convert("RGB"))
                img = cv2.cvtColor(new_img, 1)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                st.image(gray, width=500)

            # Contrast
            elif enhance_type == "Contrast":
                contrast_rate = st.sidebar.slider("Contrast Rate", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Contrast(my_image)
                img_output = enhancer.enhance(contrast_rate)
                st.image(img_output, width=500)

            # Brightness
            elif enhance_type == "Brightness":
                brightness_rate = st.sidebar.slider("Brightness Rate", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Brightness(my_image)
                img_output = enhancer.enhance(brightness_rate)
                st.image(img_output, width=500)

            elif enhance_type == "Color Balance":
                color_balance = st.sidebar.slider("Color Balance", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Color(my_image)
                img_output = enhancer.enhance(color_balance)
                st.image(img_output, width=500)
            # Blur
            elif enhance_type == "Blur":
                new_img = np.array(my_image.convert("RGB"))
                blur_rate = st.sidebar.slider("Blur Rate", 0.5, 3.0, step=0.1)
                img = cv2.cvtColor(new_img,1)
                blur_img = cv2.GaussianBlur(img, (11,11), blur_rate)
                st.image(blur_img, width=500)

            # Cartoonize
            elif enhance_type == "Cartoonize":
                result_img = cartonize_image(my_image)
                st.image(result_img, width=500)

        else:
            # Face detection
            detector_list = ["Face Detector", "Eye Detector", "Canny Edge Detector"]
            detector_choice = st.sidebar.radio("Select Detector", detector_list)
            if st.sidebar.button("Process"):
                st.subheader("Result")
                # Face detector
                if detector_choice == "Face Detector":
                    result_img, count = detect_faces(my_image)
                    st.image(result_img, width=500)
                    if count > 1:
                        st.success(f"Found {count} faces")
                    else:
                        st.success(f"Found {count} face")

                elif detector_choice == "Eye Detector":
                    result_img, count = detect_eyes(my_image)
                    st.image(result_img,  width=500)
                    if count > 1:
                        st.success(f"Found {count} eyes")
                    else:
                        st.success(f"Found {count} eye")

                elif detector_choice == "Canny Edge Detector":
                    result_img = cannize_image(my_image)
                    st.image(result_img, width=500)

if __name__ == "__main__":
    main()
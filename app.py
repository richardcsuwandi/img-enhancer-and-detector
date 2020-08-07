import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np

# Load the pre-trained Haar Cascade classifiers
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def detect_faces(image):
    img = np.array(image.convert("RGB"))
    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi = img[y:y+h, x:x+w]

        # Detect eyes in the face(s) detected
        eyes = eye_cascade.detectMultiScale(roi)

        # Detect smiles in the face(s) detected
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)

        # Draw rectangle around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        # Draw rectangle around smile
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    # Returning the image with bounding boxes drawn on it and the counts
    return img, faces

def cartoonize_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def cannize_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

def main():
    st.title("Image Enhancer and Detector 🖼️")
    st.sidebar.title("Image Enhancer and Detector 🖼️")
    st.sidebar.subheader("By [Richard Cornelius Suwandi](https://github.com/richardcsuwandi)")
    st.sidebar.markdown("[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/richardcsuwandi/img-enhancer-and-detector)")
    st.subheader("An image enhancer and detector app using PIL and OpenCV.")
    st.sidebar.markdown("An image enhancer and detector app using PIL and OpenCV.")

    image_file = st.sidebar.file_uploader("Upload image", type=["jpg","png","jpeg"])

    task = ["Image Enhancement", "Image Detection"]
    choice = st.sidebar.selectbox("Choose task", task)

    # Open and preview original image
    if image_file is not None:
        image = Image.open(image_file)
        st.subheader("Original")
        st.image(image, width=500)

        # Image enhancement
        if choice == "Image Enhancement":
            st.subheader("Result")
            types = ["Gray-Scale", "Contrast", "Brightness", "Color Balance", "Blur", "Cartoonize"]
            enhance_type = st.sidebar.radio("Enhancement Type", types)

            # Gray-scale
            if enhance_type == "Gray-Scale":
                new_img = np.array(image.convert("RGB"))
                img = cv2.cvtColor(new_img, 1)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                st.image(gray, width=500)

            # Contrast
            elif enhance_type == "Contrast":
                contrast_rate = st.sidebar.slider("Contrast Rate", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Contrast(image)
                img_output = enhancer.enhance(contrast_rate)
                st.image(img_output, width=500)

            # Brightness
            elif enhance_type == "Brightness":
                brightness_rate = st.sidebar.slider("Brightness Rate", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Brightness(image)
                img_output = enhancer.enhance(brightness_rate)
                st.image(img_output, width=500)

            # Color balance
            elif enhance_type == "Color Balance":
                color_balance = st.sidebar.slider("Color Balance", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Color(image)
                img_output = enhancer.enhance(color_balance)
                st.image(img_output, width=500)

            # Blur
            elif enhance_type == "Blur":
                new_img = np.array(image.convert("RGB"))
                blur_rate = st.sidebar.slider("Blur Rate", 0.5, 3.0, step=0.1)
                img = cv2.cvtColor(new_img,1)
                blur_img = cv2.GaussianBlur(img, (11,11), blur_rate)
                st.image(blur_img, width=500)

            # Cartoonize
            elif enhance_type == "Cartoonize":
                result_img = cartoonize_image(image)
                st.image(result_img, width=500)

        else:
            # Image detection
            detector_list = ["Face Detector", "Canny Edge Detector"]
            detector_choice = st.sidebar.radio("Select Detector", detector_list)
            if st.sidebar.button("Process"):
                st.subheader("Result")

                # Face detector
                if detector_choice == "Face Detector":
                    result_img, result_faces = detect_faces(image)
                    st.image(result_img, width=500)
                    if len(result_faces)> 1:
                        st.success(f"Found {len(result_faces)} faces")
                    else:
                        st.success(f"Found {len(result_faces)} face")

                # Canny Edge Detector
                elif detector_choice == "Canny Edge Detector":
                    result_img = cannize_image(image)
                    st.image(result_img, width=500)

    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
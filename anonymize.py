
from math import floor
import os
import numpy as np
import cv2
from ultralytics import YOLO
#from roboflow import Roboflow
import imutils
import pytesseract

    

# Source https://medium.com/artificialis/how-to-blur-faces-in-images-with-python-1275bb10ad8d
def anonymize_face(img_path):
    image = cv2.imread(img_path)

    h, w = image.shape[:2]

    kernel_width = (w//7) | 1
    kernel_height = (h//7) | 1

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    faces = face_cascade.detectMultiScale(image, 1.1, 5)

    for x, y, w, h in faces:
        face_roi = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_width, kernel_height), 0)
        image[y:y+h, x:x+w] = blurred_face

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#https://github.com/MuhammadMoinFaisal/Automatic_Number_Plate_Detection_Recognition_YOLOv8
def anonymize_plates(img_path):
    model = YOLO("best.pt")
    plates = model(img_path, retina_masks=True)
    plates = list(plates)[0]
    print(plates)

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    kernel_width = (w//7) | 1
    kernel_height = (h//7) | 1

    

    for x, y, w, h, conf, idk in plates:
        y = int(floor(y))
        h = int(floor(h))
        x = int(floor(x))
        w = int(floor(w))
        h = h - y
        w = w - x
        print(h, w)
        #face_roi = image[y:y+y2, x:x+x2]
        face_roi = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_width, kernel_height), 0)
        image[y:y+h, x:x+w] = blurred_face

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def platev2(img_path):
    rf = Roboflow(api_key="HwD3HMb11bPKMCCUNOU7")
    project = rf.workspace().project("anpr_project_merge")
    model = project.version(1).model

    # infer on a local image
    print(model.predict(img_path, confidence=40, overlap=30).json())

    # visualize your prediction
    # model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")
    

def main():
    anonymize_face("images/Sweden_001693.jpg")
    #anonymize_plates("images/swe_test.jpg")
    #platev2("images/volvo.jpg")


if __name__ == "__main__":
    main()

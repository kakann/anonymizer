
from math import floor
import os
import numpy as np
import cv2

from roboflow import Roboflow

import pytesseract

import sys
sys.path.append("/home/martin/Desktop/anonymizer/yoloface")
from face_detector import YoloDetector
from PIL import Image
import pytesseract
import time



face_counter = 0
plate_counter = 0


def anonymize(objects, image, show=False):
    h, w = image.shape[:2]
    kernel_width = (w//7) | 1
    kernel_height = (h//7) | 1
    for x, y, w, h in objects:
        y = int(floor(y))
        h = int(floor(h))
        x = int(floor(x))
        w = int(floor(w))
        #print(x, y, w, h)
        face_roi = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_width, kernel_height), 0)
        image[y:y+h, x:x+w] = blurred_face

    if show:
        cv2.imshow('image', image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    return image


def face_anonymize(img_path, model):
    orgimg = np.array(Image.open(img_path))
    bboxes,points = model.predict(orgimg, conf_thres = 0.2)
    image = cv2.imread(img_path)
    faces = []
    for x1, y1, x2, y2 in bboxes[0]:
        faces.append((x1, y1, y2 - y1, x2 - x1))
        global face_counter
        face_counter += 1
    
    return faces
    







def plate_anonymize(img_path, model):
    image = cv2.imread(img_path)
    # infer on a local image
    prediction = model.predict(img_path, confidence=20, overlap=30)
    #print(prediction)

    plates = []
    for plate in prediction:
        if plate["x"] != 0  and plate["y"] != 0 and plate["width"] != 0 and plate["height"] != 0:
            plates.append((plate["x"] - plate["width"]/2, plate["y"] - plate["height"]/2, plate["width"], plate["height"]))
            global plate_counter
            plate_counter+=1
    return plates

    # visualize your prediction
    # model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")
def name_anonymize(img_path):
    # Load the image
    image = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make the text white and the background black
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove noise using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Perform OCR on the image
    names = pytesseract.image_to_string(opening)
    if names.strip():
        return names

    # Print the extracted names
    #print(names)
    return []

def main():
    #folder to grab picures from  
    input_folder = sys.argv[1]
    target_folder = sys.argv[2]
    
    #initialize counters
    image_count = 0

    #load plate detector model
    rf = Roboflow(api_key="HwD3HMb11bPKMCCUNOU7")
    project = rf.workspace().project("anpr_project_merge")
    model_p = project.version(1).model

    #load face detector model
    model_f = YoloDetector(target_size=720, device="cuda:0", min_face=10)

    
    
    error_imgs = []
    face_imgs = []
    plate_imgs = []
    names = []
    errors = 0
    totaltime = 0
    cur_time = 0
    for i, filename in enumerate(os.listdir(input_folder)):
        cur_time = time.perf_counter()
        current_file = os.path.join(input_folder, filename)
        print(current_file)
        plates = plate_anonymize(current_file, model=model_p)
        faces = []
        try:
            faces = face_anonymize(current_file, model=model_f)
        except:
            errors+=1
            error_imgs.append(current_file)
            pass
        image = cv2.imread(current_file)

        if len(plates) > 0:
            plate_imgs.append(current_file)
        if len(plates) > 0:
            face_imgs.append(current_file)
        anonymized_image = image

        if len(faces + plates) > 0:
            anonymized_image = anonymize(faces + plates, image=image)
        cv2.imwrite(os.path.join(target_folder, filename), anonymized_image)
        
        names.append(name_anonymize(current_file))
        os.remove("/home/martin/Desktop/anonymizer/sweden/images/" + filename)
        
        image_count +=1
        totaltime += time.perf_counter() - cur_time
        print(f"Elapsed time: {totaltime} \n Average Time: {totaltime/(i+1)} \n Current Index {i+1}")

    global face_counter
    print(f"{face_counter} faces were found amount {image_count} images.")
    global plate_counter
    print(f"{plate_counter} plates were found amount {image_count} images.")
    print(f"{errors} errors were found amount {image_count} images.")
    print(names)
    print(f"Images where atleast one face has been found: \n {face_imgs}")
    print(f"Images where atleast one plate has been found: \n {plate_imgs}")
    print(f"Images where atleast one error has occured: \n {error_imgs}")

if __name__ == "__main__":
    main()

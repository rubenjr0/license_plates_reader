import cv2
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import pytesseract
import numpy as np

OCR_CONFIG = r"--oem 3 --psm 7"
EM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)



def detect_plates(image, model):
    detections = model(Image.fromarray(image))
    boxes = []
    for detection in detections:
        xmin, ymin, xmax, ymax = detection["box"].values()
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def crop_plate(image, box):
    xmin, ymin, xmax, ymax = box
    cropped = image[ymin:ymax, xmin:xmax]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("data/cropped.png", cropped)
    # upsample and denoise with median filter
    cropped = cv2.resize(cropped, (0, 0), fx=3, fy=3)
    cropped = cv2.medianBlur(cropped, 5)
    cv2.imwrite("data/denoised.png", cropped)
    return cropped

def preprocess_plate_image(image):
    # flattened = image.reshape((-1, 1))
    # flattened = np.float32(flattened)
    # em = cv2.ml.EM_create()
    # em.setClustersNumber(2)
    # em.setCovarianceMatrixType(0)
    # em.setTermCriteria(EM_CRITERIA)
    # _, _, labels, _ = em.trainEM(flattened)
    # binarized = labels.reshape(image.shape)
    # # binarized (from 0 to 1) to uint8 (from 0 to 255)
    # binarized = np.uint8(binarized * 255)
    _, binarized = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #save
    cv2.imwrite("data/plate.png", binarized)
    return binarized

def read_plate(image):
    plate = pytesseract.image_to_string(image, config=OCR_CONFIG)
    plate = plate.replace(" ", "")
    return plate

def get_plates(image, model):
    boxes = detect_plates(image, model)
    print(boxes)
    plates = []
    for box in boxes:
        plate = crop_plate(image, box)
        plate = preprocess_plate_image(plate)
        plate = read_plate(plate)
        plates.append(plate)
    return plates

pipe = pipeline(
    "object-detection",
    model="nickmuchi/yolos-small-finetuned-license-plate-detection",
)

img_original = cv2.imread("data/car_2.png")
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

plates = get_plates(img_original, pipe)
for plate in plates:
    print(plate)
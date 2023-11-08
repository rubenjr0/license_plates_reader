import sys
import cv2
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import pytesseract
import numpy as np

# TODO: Improve the OCR
OCR_CONFIG = r"--oem 3 --psm 7"
EM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)


def detect_plates(image, model):
    detections = model(Image.fromarray(image))
    boxes = []
    for detection in detections:
        xmin, ymin, xmax, ymax = detection["box"].values()
        score = detection["score"]
        boxes.append((xmin, ymin, xmax, ymax))
        # draw the bounding boxes
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(
            image,
            f"{score:.2f}",
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    # save in RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("data/process/detected.png", image)
    return boxes


def crop_plate(image, box):
    xmin, ymin, xmax, ymax = box
    # TODO: Crop the license plate
    # xmin = int(xmin * 1.08)
    # xmax = int(xmax * 0.98)
    # ymin = int(ymin * 1.02)
    # ymax = int(ymax * 0.99)
    cropped = image[ymin:ymax, xmin:xmax]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("data/process/cropped.png", cropped)
    # upsample and denoise with median filter
    cropped = cv2.resize(cropped, (0, 0), fx=4, fy=4)
    return cropped


def preprocess_plate_image(image):
    # normalize and binarize
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("data/process/normalized.png", image)
    flattened = image.reshape((-1, 1))
    flattened = np.float32(flattened)
    em = cv2.ml.EM_create()
    em.setClustersNumber(2)
    em.setCovarianceMatrixType(0)
    em.setTermCriteria(EM_CRITERIA)
    _, _, labels, _ = em.trainEM(flattened)
    binarized = labels.reshape(image.shape)
    # binarized (from 0 to 1) to uint8 (from 0 to 255)
    binarized = np.uint8(binarized * 255)
    # _, binarized = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    # save
    cv2.imwrite("data/process/binarized.png", binarized)
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

img_path = sys.argv[1]
img_original = cv2.imread(img_path)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

plates = get_plates(img_original, pipe)
for plate in plates:
    print(plate)

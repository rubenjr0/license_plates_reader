# License plate recognition
## Pipeline
- Detect plates with a finetuned YOLO model
- Crop the regions of interest
- Preprocess
- Perform OCR

## Demo
We first load a demo image:
![Demo image](data/car_2.jpg)

We detect all plates:
![Detected plates](data/process/detected.png)

We crop the region of interest:

![Cropped plate](data/process/cropped.png)

We scale, normalize, and binarize:
![Normalized](data/process/normalized.png)
![Binarized](data/process/binarized.png)

### Results
> Detected: `51F069.489`

# TODO
- Improve cropping
- Improve OCR
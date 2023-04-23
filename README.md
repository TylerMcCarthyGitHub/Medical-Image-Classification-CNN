# Medical-Image-Classification-CNN
An implementation of the scientific paper  [Medical Image Classification with Convolutional Neural Networks](https://ieeexplore.ieee.org/iel7/7056237/7064265/07064414.pdf?casa_token=UXUPIF9mudIAAAAA:xtEddGglzB-nGlTIjH0UrM3orRgYjjWlqYIowWbj8oNdQWsQmDPUdy9hvGWsqFx9htejav_E).

## Data
### Original source:
The Open access chest X-ray collection from Indiana University was obtained from the National Library of Medicine's Open-i service (https://openi.nlm.nih.gov/). The original images were in DICOM format and were collected from various sources, including hospitals and research studies.

### Current form:
The images were preprocessed and converted to PNG format by clipping the top/bottom 0.5% of pixel values to eliminate outliers, linearly scaling the pixel values to fit into a 0-255 range, and resizing to 2048 pixels on the shorter side to fit into the Kaggle dataset limits. Each image was manually classified into frontal and lateral chest X-ray categories. The metadata for each image was downloaded using the available Open-i API.

The dataset also includes a CSV file containing the unique identifiers (UIDs) for each image, along with additional information such as MeSH terms, clinical problems, indications for the imaging studies, comparisons to prior studies, and findings. This information was obtained from the original DICOM files and/or from radiology reports associated with the images.

## Project Structure
```
.
├── README.md
├── config.yaml
├── data
│   └── 
│       ├── 0001.png
│       ├── ...
│       └── 0900.png
├── notebooks
│   └── Inference.ipynb
├── requirements.txt
├── train.py
├── utils
│   ├── constants.py
│   ├── dataset.py
│   └── model.py
└── weights
    └── model.h5
```

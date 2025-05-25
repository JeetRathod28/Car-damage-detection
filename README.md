# Car Damage Detection

A dual-model project using YOLOv8 and TensorFlow to detect car damage from images and videos.

## Project Structure

car-damage-detection/
├── models/
│   ├── yolo_best.pt            # Pretrained YOLOv8 model
│   └── damage_classifier.h5    # Pretrained TensorFlow model (download required, see below)
├── samples/
│   ├── images/                 # Sample test images
│   └── videos/                 # Sample test videos
├── run.ipynb                   # Jupyter Notebook with inference and demo pipeline
├── train/
│   ├── train_yolo.py           # YOLOv8 training script
│   └── train_tf.py             # TensorFlow training script
├── requirements.txt            # Required Python packages
└── README.md                   # This file

## Models

- **YOLOv8** (`best.pt`): Detects visible car damage.
- **TensorFlow** (`damage_classifier.h5`): Classifies type/severity of damage from the detected regions.


## Model Download
Due to file size limitations on GitHub, the pretrained TensorFlow model (damage_classifier.h5) is hosted on Google Drive. Download it from the following link:
https://drive.google.com/file/d/1iAamXRinh7vjbW7NGp_IcBIWLaig1LJX/view?usp=sharing 
Download TensorFlow Model
After downloading, place the damage_classifier.h5 file in the models/ directory of the project.

## Training Code Provided

Training code is provided separately in the `train/` directory:

- `train/train_yolo.py`: For training the YOLOv8 model. Requires annotated images in YOLO format.
- `train/train_tf.py`: For training the TensorFlow classifier. Requires cropped images and class labels.

> ⚠️ **Important Notes:**  
> - **Training requires a GPU.** Do **NOT** attempt to train models on a CPU — it will be extremely slow or may crash.  
> - You must have the proper dataset, including both images and labeled annotations (bounding boxes for YOLO, categories for TensorFlow), to train the models.

## How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/JeetRathod28/Car-damage-detection.git
   cd Car-damage-detection
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Download the TensorFlow model:
- Follow the instructions in the Model Download section to download `damage_classifier.h5`.
- Place it in the models/ directory.
  
4. Run the inference/demo code:
- Open `main.ipynb` in Jupyter Notebook.
- Update the paths to your image, video, or dataset files.
- Run all cells — everything is modular and works directly after setting paths.

🧪 Test Samples
Image: samples/images/car2.jpg
Video: samples/video.mp4

All required Python packages are listed in requirements.txt.

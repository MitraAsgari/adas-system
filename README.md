# Advanced Driver Assistance System (ADAS)

Developed an Advanced Driver Assistance System to detect and prevent accidents using the Udacity Self-Driving Car Dataset. Preprocessed data by extracting and normalizing video frames. Trained CNNs for object detection and developed models for path prediction and decision-making. Evaluated using IoU and mAP metrics. Simulated the system in CARLA for realistic testing.

## Features
- Video frame extraction and normalization
- CNN training for object detection
- Path prediction and decision-making models
- Simulation in CARLA

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/adas-system.git
2. Navigate to the project directory:
   ```bash
   cd adas-system
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
## Usage
1. Run the data preprocessing script:
   ```bash
   python preprocess.py
2. Train the models:
   ```bash
   python train.py
3. Simulate the system in CARLA:
   ```bash
   python simulate.py

## Prerequisites
Python 3.x 
Libraries: numpy, pandas, tensorflow, opencv-python, carla

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT

## Link to Project Code
https://github.com/MitraAsgari/adas-system

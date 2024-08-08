import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Extract bounding boxes from XML files
def extract_boxes(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = []
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

# Load image files and split them into training and test sets
def load_data(data_dir):
    image_files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
    return train_files, test_files

if __name__ == "__main__":
    data_dir = '/content/data/export'
    train_files, test_files = load_data(data_dir)
    print(f"Number of training files: {len(train_files)}")
    print(f"Number of test files: {len(test_files)}")

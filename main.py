import numpy as np
import cv2
from brisque import *
import os

# Coffee mug
"Images/KonIQ-10k/512x384/2704811.jpg"
# Small dog with red jacket
"Images/KonIQ-10k/512x384/5354698.jpg"

if __name__=="__main__":
    path = "Images/KonIQ-10k/512x384"
    files = os.listdir(path)
    print(files)
    #image = cv2.imread(f"{path}/{files[6]}")

    for file in files:
        image = cv2.imread(f"{path}/{file}")
        GetBrisqueScore(image)
        stop

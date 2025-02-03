'''
This script takes a pair of image/label and applies the following preprocessing:
Progressively increase/decrease the image size N times, and save the resized 
image/label pairs. This produces 2N image/label pairs.

Author: Armand Collin
'''

from pathlib import Path
import argparse
import numpy as np
from AxonDeepSeg.ads_utils import imread, imwrite

def main()
    ...

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image")
    ap.add_argument("-g", "--gt", required=True, help="path to input GT label")
    ap.add_argument("-n", "--num-samples", required=True, help="number of times to resize the image") 
    ap.add_argument("-l", "--lower", default=0.5, help="lower bound of the resizing factor")
    ap.add_argument("-u", "--upper", default=3.0, help="upper bound of the resizing factor")
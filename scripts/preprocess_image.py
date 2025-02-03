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
from skimage.transform import resize

def get_sizes(image, lower, upper, N):
    # the first half of the sizes are decreasing, the second half increasing
    sizes = []
    for i in range(N//2):
        factor_decreasing = 1 - (i+1) * (1 - lower) / (N//2)
        factor_increasing = 1 + (i+1) * (upper - 1) / (N//2)
        sizes.append((int(image.shape[0] * factor_decreasing), int(image.shape[1] * factor_decreasing)))
        sizes.append((int(image.shape[0] * factor_increasing), int(image.shape[1] * factor_increasing)))
    return sizes

def main(input, gt, num_samples, lower, upper):
    img = imread(input)
    sizes = get_sizes(img, lower, upper, num_samples)
    print(img.shape)
    print(sizes)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image")
    ap.add_argument("-g", "--gt", required=True, help="path to input GT label")
    ap.add_argument("-n", "--num-samples", required=True, help="number of times to resize the image") 
    ap.add_argument("-l", "--lower", default=0.5, help="lower bound of the resizing factor")
    ap.add_argument("-u", "--upper", default=3.0, help="upper bound of the resizing factor")

    args = ap.parse_args()
    main(args.input, args.gt, int(args.num_samples), args.lower, args.upper)
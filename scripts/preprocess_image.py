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

def get_sizes_from_directory(dir):
    '''we assume that dir contains the resized images (_expansion/imgs/)'''
    sizes = []
    for fname in dir.glob('*.png'):
        if 'im_' in fname.stem:
            sizes.append(tuple(map(int, fname.stem.split('_')[1:])))
    return sorted(sizes)

def get_sizes(image, lower, upper, N):
    # the first half of the sizes are decreasing, the second half increasing
    sizes = []
    for i in range(N//2):
        factor_decreasing = 1 - (i+1) * (1 - lower) / (N//2)
        factor_increasing = 1 + (i+1) * (upper - 1) / (N//2)
        sizes.append((int(image.shape[0] * factor_decreasing), int(image.shape[1] * factor_decreasing)))
        sizes.append((int(image.shape[0] * factor_increasing), int(image.shape[1] * factor_increasing)))
    sizes.append(image.shape)
    return sorted(sizes)

def expand_image(impath, num_samples, lower=0.25, upper=8.0):
    '''output file structure is as follows:
        IMAGE.png
        IMAGE_seg-axon-manual.png
        IMAGE_seg-myelin-manual.png
        IMAGE_expansion                 <<< output_dir
        ├── imgs
        │   ├── im_256_256.png
        │   ├── ...
        ├── gts
        │   ├── gt-axon_256_256.png
        │   ├── gt-myelin_256_256.png
        │   ├── ...
        ├── preds
        │   ├── model-1
        │   │   ├── im_256_256_seg-axon.png
        │   │   ├── im_256_256_seg-myelin.png
        │   │   ├── ...
        │   ├── ...
    '''
    img = imread(impath)
    axon_path = impath.replace(".png", "_seg-axon-manual.png")
    myelin_path = impath.replace(".png", "_seg-myelin-manual.png")
    axon_gt = imread(axon_path)
    myelin_gt = imread(myelin_path)
    sizes = get_sizes(img, lower, upper, num_samples)

    print(f'Source shape: {img.shape}')
    print(f'The image will be resized to the following target shapes: {sizes}')

    output_dir = Path(impath).parent / f'{Path(impath).stem}_expansion'
    output_dir.mkdir(exist_ok=True)
    imgs_output_dir = output_dir / 'imgs'
    gts_output_dir = output_dir / 'gts'
    imgs_output_dir.mkdir(exist_ok=True)
    gts_output_dir.mkdir(exist_ok=True)

    for size in sizes:
        new_image = resize(img, size, preserve_range=True)
        new_axon_gt = resize(axon_gt, size, order=0, preserve_range=True, anti_aliasing=False)
        new_myelin_gt = resize(myelin_gt, size, order=0, preserve_range=True, anti_aliasing=False)
        
        imwrite(imgs_output_dir / f'im_{size[0]}_{size[1]}.png', new_image)
        imwrite(gts_output_dir / f'gt-axon_{size[0]}_{size[1]}.png', new_axon_gt)
        imwrite(gts_output_dir / f'gt-myelin_{size[0]}_{size[1]}.png', new_myelin_gt)

def main(impath, num_samples, lower, upper):
    expand_image(impath, num_samples, lower, upper)
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--impath", required=True, help="path to impath image. We assume the corresponding labels are in the same directory and have the same name with suffixes '_seg-axon-manual' and '_seg-myelin-manual'")
    ap.add_argument("-n", "--num-samples", required=True, help="number of times to resize the image")
    ap.add_argument("-l", "--lower", default=0.25, help="lower bound of the resizing factor")
    ap.add_argument("-u", "--upper", default=4.0, help="upper bound of the resizing factor")

    args = ap.parse_args()
    main(args.impath, int(args.num_samples), args.lower, args.upper)

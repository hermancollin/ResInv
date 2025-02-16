'''
This script is used to evaluate the stability of the model under resolution 
perturbations. A given model is applied to a range of resized images, this script 
will compute the Dice coefficient in the native referential (ground truth VS resized 
prediction) and in the resized referential (resized ground truth VS prediction).

Author: Armand Collin
'''

from preprocess_image import get_sizes_from_directory, expand_image
from plot_results import plot_and_write_results, plot_model_comparison

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from monai.metrics import DiceMetric
from AxonDeepSeg.ads_utils import imread
from AxonDeepSeg.segment import segment_images
from skimage.transform import resize


    #===============#
    # visualization #
    #===============#



    #=============#
    # apply model #
    #=============#
def apply_model(model, impath):
    # verify if image is already expanded
    expansion_dir = Path(impath).parent / f'{Path(impath).stem}_expansion'
    if expansion_dir.exists():
        print('Image already expanded - skipping expansion.')
    else:
        expand_image(impath=impath, num_samples=20)
    
    im_paths = list((expansion_dir / 'imgs').glob('*.png'))
    device = 0 if torch.cuda.is_available() else -1
    
    segment_images(
        path_images=im_paths,
        path_model=model,
        gpu_id=device,
    )
    print(f'Model {Path(model).stem} applied to all images in expansion.')
    # move all predictions to the right directory
    preds_dir = expansion_dir / 'preds' / Path(model).stem
    preds_dir.mkdir(parents=True, exist_ok=True)
    for pred_path in (expansion_dir / "imgs").glob(f'*_seg-*.png'):
        pred_path.replace(preds_dir / pred_path.name)
    # delete all _seg-axonmyelin.png imgs (not needed)
    for pred in preds_dir.glob('*_seg-axonmyelin.png'):
        pred.unlink()

    #============#
    # evaluation #
    #============#
def convert_numpy_to_tensor(img):
    img = np.where(img > 0, 1, 0)
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    return tensor

def load_binary_img_as_tensor(img_path):
    img = imread(img_path)
    return convert_numpy_to_tensor(img)

def evaluate(preds_path, img_path):
    '''Computes Dice coefficient, returns a pandas DataFrame with the following columns:
        - Image Height
        - Image Width
        - Resize Factor
        - Dice coefficient in the native referential (for both axon and myelin)
        - Dice coefficient in the resized referential (for both axon and myelin)
        - Interpolation error (difference between gt and resized-gt)
    '''

    cols = [
        'Image Height', 'Image Width', 'Resize Factor', 'Dice Axon Native', 
        'Dice Myelin Native', 'Dice Axon Resized', 'Dice Myelin Resized', 
        'Dice Axon Interpolation', 'Dice Myelin Interpolation'
    ]
    results = pd.DataFrame(columns=cols)
    imgs_path = Path(img_path).parent / f'{Path(img_path).stem}_expansion/imgs'
    gts_path = Path(img_path).parent / f'{Path(img_path).stem}_expansion/gts'
    sizes = get_sizes_from_directory(imgs_path)
    native_h, native_w = imread(img_path).shape
    for size in sizes:
        h, w = size
        resize_factor = h / native_h

        # load GTs
        gt_axon = load_binary_img_as_tensor(img_path.replace('.png', f'_seg-axon-manual.png'))
        gt_myelin = load_binary_img_as_tensor(img_path.replace('.png', f'_seg-myelin-manual.png'))
        gt_axon_resized = load_binary_img_as_tensor(gts_path / f'gt-axon_{h}_{w}.png')
        gt_myelin_resized = load_binary_img_as_tensor(gts_path / f'gt-myelin_{h}_{w}.png')

        # load predictions
        pred_axon_resized = load_binary_img_as_tensor(preds_path / f'im_{h}_{w}_seg-axon.png')
        pred_myelin_resized = load_binary_img_as_tensor(preds_path / f'im_{h}_{w}_seg-myelin.png')

        dice_metric = DiceMetric(include_background=False, reduction='mean')
        # compute Dice in shifted referential
        dice_axon_resized = dice_metric([pred_axon_resized], [gt_axon_resized]).item()
        dice_myelin_resized = dice_metric([pred_myelin_resized], [gt_myelin_resized]).item()

        # compute Dice in native referential
        pred_axon_resized_back = imread(preds_path / f'im_{h}_{w}_seg-axon.png')
        pred_axon_resized_back = resize(pred_axon_resized_back, (native_h, native_w), order=0, preserve_range=True, anti_aliasing=False)
        pred_axon_resized_back = convert_numpy_to_tensor(pred_axon_resized_back)
        pred_myelin_resized_back = imread(preds_path / f'im_{h}_{w}_seg-myelin.png')
        pred_myelin_resized_back = resize(pred_myelin_resized_back, (native_h, native_w), order=0, preserve_range=True, anti_aliasing=False)
        pred_myelin_resized_back = convert_numpy_to_tensor(pred_myelin_resized_back)
        dice_axon_native = dice_metric([pred_axon_resized_back], [gt_axon]).item()
        dice_myelin_native = dice_metric([pred_myelin_resized_back], [gt_myelin]).item()

        # compute dice of interpolation
        gt_axon_resized_back = imread(gts_path / f'gt-axon_{h}_{w}.png')
        gt_axon_resized_back = resize(gt_axon_resized_back, (native_h, native_w), order=0, preserve_range=True, anti_aliasing=False)
        gt_axon_resized_back = convert_numpy_to_tensor(gt_axon_resized_back)
        gt_myelin_resized_back = imread(gts_path / f'gt-myelin_{h}_{w}.png')
        gt_myelin_resized_back = resize(gt_myelin_resized_back, (native_h, native_w), order=0, preserve_range=True, anti_aliasing=False)
        gt_myelin_resized_back = convert_numpy_to_tensor(gt_myelin_resized_back)
        dice_axon_interpolation = dice_metric([gt_axon_resized_back], [gt_axon]).item()
        dice_myelin_interpolation = dice_metric([gt_myelin_resized_back], [gt_myelin]).item()

        new_row = {
            'Image Height': h, 'Image Width': w, 'Resize Factor': resize_factor, 
            'Dice Axon Native': dice_axon_native, 'Dice Myelin Native': dice_myelin_native, 
            'Dice Axon Resized': dice_axon_resized, 'Dice Myelin Resized': dice_myelin_resized, 
            'Dice Axon Interpolation': dice_axon_interpolation, 'Dice Myelin Interpolation': dice_myelin_interpolation
        }
        results.loc[len(results)] = new_row

    # save results in CSV
    print('Saving results.')
    results.to_csv(preds_path / 'evaluation.csv', index=False)


def main(impath, modelpath, plot_only):
    if modelpath:
        apply_model(modelpath, impath)

    # predictions are in _expansion/preds/
    preds_dir = Path(impath).parent / f'{Path(impath).stem}_expansion/preds'
    assert preds_dir.exists(), f'No predictions found in {preds_dir}'

    models = [d.name for d in preds_dir.glob('*') if d.is_dir()]
    for model in models:
        if modelpath and model != Path(modelpath).name:
            continue
        model_preds_path = preds_dir / model
        if not plot_only:
            print(f'Evaluation for model {model}')
            evaluation = evaluate(model_preds_path, impath)
        print(f'Plotting results for model {model}.')
        plot_and_write_results(model_preds_path)

    if len(models) > 1:
        print('Plotting model comparison.')
        plot_model_comparison(preds_dir)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--impath", required=True, help="path to original image")
    ap.add_argument("-m", "--model", default=None, required=False, help="path to model")
    ap.add_argument("-p", "--plot", action='store_true', help='only plot existing results')

    args = ap.parse_args()
    main(args.impath, args.model, args.plot)

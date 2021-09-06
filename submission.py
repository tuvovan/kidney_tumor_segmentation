import os
from PIL import Image
import argparse
import numpy as np
from tqdm import tqdm

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def make_submission(input_dir, save_path, num_classes):
    f = open(save_path, "w")
    f.write("Id,EncodedPixels\n")
    case_names = sorted(os.listdir(input_dir))
    for case_name in tqdm(case_names):
        case_dir = os.path.join(input_dir, case_name)
        slice_names = sorted(os.listdir(case_dir))
        for class_idx in range(1, num_classes):
            mask_stack = np.array([], dtype='uint8')
            for slice_name in slice_names:
                slice_img_path = os.path.join(case_dir, slice_name)
                slice_img = np.array(Image.open(slice_img_path).convert('L')).flatten()
                slice_mask = np.equal(slice_img, class_idx) * 255
                mask_stack = np.hstack([mask_stack, slice_mask])
            data_id = f'{case_name}_{class_idx}'
            enc = rle_to_string(rle_encode(mask_stack))
            line = f'{data_id},{enc}'
            f.write(line + "\n")
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', required=False, type=str,
                        help="Directory where labels are located",
                        default='test_results_softmax/',
                        )
    parser.add_argument('-save_path', required=False, default='./submission_image_sm_2mds.csv', type=str,
                        help="submission csv file to save")
    parser.add_argument('-num_classes', required=False, default=3, type=int,
                        help="number of classes")
    args = parser.parse_args()
    make_submission(input_dir=args.input_dir,
                    save_path=args.save_path,
                    num_classes=args.num_classes)

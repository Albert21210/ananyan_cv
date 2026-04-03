import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def classificator(region):
    holes = count_holes(region)
    if holes == 2: #B, 8
        labeled_inv = label(np.logical_not(region.image))
        bays_left = 0
        for r in regionprops(labeled_inv):
            if r.bbox[1] == 0 and r.area > 2:
                bays_left += 1
        
        if bays_left > 1: 
            return "8"
        else:
            return "B"
    
    elif holes == 1: # A, 0
        height, width = region.image.shape        
        bottom_strip = region.image[-max(1, int(height * 0.1)): , :]
        
        center_width = int(width * 0.4)
        center_start = (width - center_width) // 2
        center_part = bottom_strip[:, center_start : center_start + center_width]
        
        bottom_filling = np.mean(center_part)
        
        if bottom_filling < 0.2:
            return "A"
        else:
            return "0"
        
    else: # 1, W, X, *, /, -
        if region.image.sum() / region.image.size == 1:
            return "-"
        
        shape = region.image.shape
        aspect = np.min(shape) / np.max(shape)

        if aspect > 0.85:
            return "*"

        labeled_inv = label(np.logical_not(region.image))
        bays = 0
        for r in regionprops(labeled_inv):
            if r.area > 2:
                bays += 1

        if bays == 5:
            return "W"
        elif bays == 4:
            return "X"
        elif bays == 2:
        
            v_sums = np.sum(region.image, axis=0)
            v_line_ratio = np.max(v_sums) / shape[0]
            
            if v_line_ratio > 0.8: 
                return "1"
            else:
                return "/"
        
        v_sums = np.sum(region.image, axis=0)
        if np.max(v_sums) / shape[0] > 0.8:
            return "1"
            
    return "?"


image = imread("alphabet.png")[:, :, :-1]
abinary = image.mean(2) > 0
alphabet = label(abinary)
aprops = regionprops(alphabet)

result = {}

image_path = save_path / "out_tree"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5, 7))

for region in aprops:
    symbol = classificator(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print(result)
print(f"Процент распознавания: {(1 - result.get("?", 0) / len(aprops)) * 100}")

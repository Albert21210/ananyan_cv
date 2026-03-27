import numpy as np
from skimage.measure import label

image = np.load("stars.npy")

labeled = label(image)

plus = 0
cross = 0

for i in range(1, labeled.max() + 1):
    mask = (labeled == i)

    if mask.sum() == 9:
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        if (y_max - y_min + 1) == 5:
            if image[y_min, x_min] == 1:
                cross += 1
            else:
                plus += 1

print(f"Количество плюсов: {plus}")
print(f"Количество крестов: {cross}")
print(f"Количество звездочек: {plus + cross}")

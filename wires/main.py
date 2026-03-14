import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import opening

image = np.load("wires/wires3.npy")
struct = np.ones((3,1))
process = opening(image, struct)

labeled_image = label(image)
labeled_process = label(process)

count = 0

for i in range(1, np.max(labeled_image) + 1):
    wire = labeled_image == i
    wire = opening(wire, struct)
    labeled_wire = label(wire)
    count = np.max(labeled_wire)
    
    print(f"Труба {i} | Количество: {count} ")

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(opening(image, struct))
plt.show()

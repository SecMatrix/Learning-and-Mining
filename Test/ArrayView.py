import numpy as np
from PIL import Image

def display_image(x):
    x_scaled = np.uint8(255 * (x - x.min()) / (x.max() - x.min()))
    return Image.fromarray(x_scaled)

display_image(np.random.rand(1000,1000)).show()
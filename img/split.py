import os

from PIL import Image
import numpy as np

def split(name):
    
    im_name = f"{name}.png"
    im = np.array(Image.open(im_name))

    print(im.shape)

    im2 = im[258:,:,:]
    print(im2.shape)

    im2 = Image.fromarray(im2)
    im2.save(f"{name}_split.png")

split("285300_0")
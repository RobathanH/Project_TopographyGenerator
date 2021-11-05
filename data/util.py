import numpy as np
import matplotlib.pyplot as plt

'''
Simple utility functions for working with this heightmap data
'''


'''
Calls plt.imshow (or ax.imshow) with the colormap set to
coolwarm, forcing the middle of the colormap (white) to 
correspond to magnitude zero in the data (water level)
'''
def fixed_zero_imshow(img, ax=None):
    max_mag = np.max(np.abs(img))
    if ax is None:
        plt.imshow(img, vmin=-max_mag, vmax=max_mag, cmap='coolwarm')
    else:
        ax.imshow(img, vmin=-max_mag, vmax=max_mag, cmap='coolwarm')

'''
Plots and saves a list of images
'''
def save_images(imgs, title=None, filename="temp.png"):
    count = imgs.shape[0]
    fig, axarr = plt.subplots(count, 1, figsize=(6, 3 * count), constrained_layout=True)
    for i in range(count):
        fixed_zero_imshow(imgs[i], axarr[i])
    
    if title is not None:
        fig.suptitle(title)
    
    plt.show()
    plt.savefig(filename)

'''
Plots and saves two lists of corresponding images side by side
'''
def save_images2(imgs1, imgs2, main_title=None, title1=None, title2=None, filename="temp.png"):
    count = min(imgs1.shape[0], imgs2.shape[0])
    fig, axarr = plt.subplots(count, 2, figsize=(9, 3 * count), constrained_layout=True)
    for i in range(count):
        fixed_zero_imshow(imgs1[i], axarr[i][0])
        fixed_zero_imshow(imgs2[i], axarr[i][1])
        if title1 is not None:
            axarr[i][0].title.set_text(title1)
        if title2 is not None:
            axarr[i][1].title.set_text(title2)

    if main_title is not None:
        fig.suptitle(main_title)

    plt.show()
    plt.savefig(filename)
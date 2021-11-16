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
        plt.axis('off')
        plt.imshow(img.T, vmin=-max_mag, vmax=max_mag, cmap='coolwarm', origin='lower')
    else:
        ax.set_axis_off()
        ax.imshow(img.T, vmin=-max_mag, vmax=max_mag, cmap='coolwarm', origin='lower')

'''
Save a multiple lists of images, where images correspond across each list
'''
def save_image_lists(imgs, titles, main_title=None, filename="temp.png"):
    column_count = min(len(imgs), len(titles))
    row_count = min(*[len(i) for i in imgs])
    fig, axarr = plt.subplots(row_count, column_count, figsize=(4 * column_count, 3 * row_count), constrained_layout=True)

    for i in range(row_count):
        for j in range(column_count):
            fixed_zero_imshow(imgs[j][i], axarr[i][j])
            axarr[i][j].title.set_text(titles[j])

    if main_title is not None:
        fig.suptitle(main_title)

    plt.show()
    plt.savefig(filename)
    plt.close('all')
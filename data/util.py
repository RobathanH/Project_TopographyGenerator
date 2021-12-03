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
Args:
    imgs ([[np.array]]):                                List of lists of np array images. First dimension is column, second is row
    titles (Function[(col, row) -> title] OR None):     List of titles for each column or a function to determine title for each image
    main_title (str OR None):                           
    filename (str):                                     
'''
def save_image_lists(imgs, titles=None, main_title=None, filename="temp.png"):
    if titles is not None and not callable(titles):
        raise ValueError(f"titles variable must be function: {str(titles)}")

    column_count = len(imgs)
    row_count = min(*[len(i) for i in imgs], len(imgs[0]))
    total_x_pixels = max(0, *[sum(imgs[col][row].shape[0] for col in range(column_count)) for row in range(row_count)]) * (1 + 0.01 * (column_count - 1))
    total_y_pixels = max(0, *[sum(imgs[col][row].shape[1] for row in range(row_count)) for col in range(column_count)]) * (1 + 0.01 * (row_count - 1))
    fig_y_len = total_y_pixels / 64 # in inches
    fig_x_len = total_x_pixels / 64
    fig, axarr = plt.subplots(row_count, column_count, figsize=(fig_x_len, fig_y_len), constrained_layout=True, squeeze=False)

    for i in range(row_count):
        for j in range(column_count):
            fixed_zero_imshow(imgs[j][i], axarr[i][j])
            if titles is not None:
                axarr[i][j].title.set_text(titles(j, i))

    if main_title is not None:
        fig.suptitle(main_title)

    plt.show()
    plt.savefig(filename)
    plt.close('all')
from .outpaint_handler import *
from ..architectures.outpaint_large_2d import *

'''
Uses the generator architecture from outpaint models and applies it to generation based on two adjacent images
'''

class OutPaint2DHandler(OutPaintHandler):
    def __init__(self, name: str, img_dims: Tuple[int, int], region_dims: Tuple[float, float], generator, local_discriminator, global_discriminator, weight_decay=0):
        super(OutPaint2DHandler, self).__init__(name, img_dims, region_dims, generator, local_discriminator, global_discriminator, weight_decay)

        # Change mask to account for generating only one quadrant
        mask = np.ones(img_dims)
        x_mid, y_mid = img_dims[0] // 2, img_dims[1] // 2
        cos_x = np.cos(np.linspace(0, np.pi / 2, img_dims[0] - x_mid, endpoint=False)).reshape(-1, 1)
        cos_y = np.cos(np.linspace(0, np.pi / 2, img_dims[1] - y_mid, endpoint=False)).reshape(1, -1)
        mask_hh = np.maximum(cos_x, cos_y)
        mask[x_mid:, y_mid:] = mask_hh
        mask = mask.reshape(1, 1, *img_dims)
        self.recon_loss_mask = torch.from_numpy(mask).to(DEVICE)

        # Change local discriminator formatter to only look at top-right quadrant
        self.local_discriminator_formatter = lambda full_image : split_quadrants(full_image)[3]
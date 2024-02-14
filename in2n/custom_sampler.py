from typing import Optional, Union

import torch
from jaxtyping import Int
from torch import Tensor

from nerfstudio.data.pixel_samplers import PatchPixelSampler

class CustomPatchPixelSampler(PatchPixelSampler):
    
    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False, **kwargs):
        super().__init__(num_rays_per_batch, keep_full_image, **kwargs)
        self.view_skip_step = 1

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        if isinstance(mask, Tensor):
            assert True, "Custom patch sampler is not implemented for datasets with masks"
        else:
            sub_bs = batch_size // (self.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.patch_size, image_width - self.patch_size],
                device=device,
            )

            indices = indices.view(sub_bs, 1, 1, 3).broadcast_to(sub_bs, self.patch_size, self.patch_size, 3).clone()

            yys, xxs = torch.meshgrid(
                torch.arange(self.patch_size, device=device), torch.arange(self.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)
            
            if self.view_skip_step > 1:
                indices[:, 0] = indices[:, 0] - torch.remainder(indices[:, 0], self.view_skip_step)

        return indices
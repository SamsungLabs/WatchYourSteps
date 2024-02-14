# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model for InstructNeRF2NeRF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, Dict, List, Union, cast, Tuple
from torch.nn import Parameter

import torch
from torch import Tensor
import torch.nn.functional as F
import copy
from collections import defaultdict
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    interlevel_loss,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    MSELoss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)

from in2n.custom_field import ErrorField, AuxField

@dataclass
class InstructNeRF2NeRFModelConfig(NerfactoModelConfig):
    """Configuration for the InstructNeRF2NeRFModel."""
    _target: Type = field(default_factory=lambda: InstructNeRF2NeRFModel)
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    use_lpips: bool = True
    """Whether to use LPIPS loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 32
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = 1.0
    """Multiplier for LPIPS loss."""
    render_aux_by_default: bool = False
    """Whether to render the auxiliary model by default"""
    render_error_by_default: bool = False
    """Whether to render the errors field by default"""

class InstructNeRF2NeRFModel(NerfactoModel):
    """Model for InstructNeRF2NeRF."""

    config: InstructNeRF2NeRFModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        self.aux_field = AuxField(
            aabb=self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )
        self.aux_field.load_state_dict(copy.deepcopy(self.field.state_dict()))
        self.aux_field.get_density = self.field.get_density

        # self.error_field = ErrorField(
        #     aabb=self.scene_box.aabb,
        #     num_levels=self.config.num_levels,
        #     max_res=self.config.max_res,
        #     log2_hashmap_size=self.config.log2_hashmap_size,
        #     hidden_dim_color=self.config.hidden_dim_color,
        #     num_images=self.num_train_data,
        #     use_pred_normals=self.config.predict_normals,
        #     use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        # )
        # self.error_field.get_density = self.field.get_density
        self.error_field = copy.deepcopy(self.field)

        # self.config.render_error_by_default = True

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        # if "error" in outputs:
        #     with torch.no_grad():
        #         error_map = torch.abs(image - outputs["rgb"])
        #     loss_dict["error_loss"] = MSELoss()(outputs["error"], error_map)
        if "error" in outputs:
            heatmap = batch["heatmap"].to(self.device)
            valid_indices = heatmap != -1
            loss_dict["error_loss"] = MSELoss()(outputs["error"][valid_indices], heatmap[valid_indices])

        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.config.use_lpips:
            out_patches = (outputs["rgb"].view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            gt_patches = (image.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * self.lpips(out_patches, gt_patches)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )

        return loss_dict
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        param_groups["aux_fields"] = list(self.aux_field.parameters())
        param_groups["error_fields"] = list(self.error_field.parameters())
        return param_groups
    
    def get_outputs(self, ray_bundle: RayBundle, render_args: Dict = {}):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        if "detach_weights" in render_args and render_args["detach_weights"]:
            weights = weights.detach()
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        if "render_aux" in render_args and render_args["render_aux"]:
            aux_field_outputs = self.aux_field.forward(ray_samples, compute_normals=self.config.predict_normals)
            if self.config.use_gradient_scaling:
                aux_field_outputs = scale_gradients_by_distance_squared(aux_field_outputs, ray_samples)
            rgb = self.renderer_rgb(rgb=aux_field_outputs[FieldHeadNames.RGB], weights=weights)
        else:
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if "render_error" in render_args and render_args["render_error"]:
            error_field_outputs = self.error_field.forward(ray_samples, compute_normals=self.config.predict_normals)
            if self.config.use_gradient_scaling:
                error_field_outputs = scale_gradients_by_distance_squared(error_field_outputs, ray_samples)
            error = self.renderer_rgb(rgb=error_field_outputs[FieldHeadNames.RGB][:, :, :1], weights=weights.detach())

            outputs["error"] = error

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def forward(self, ray_bundle: RayBundle, render_args: Dict = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        if render_args is None and not self.training:
            render_args = {"render_aux": self.config.render_aux_by_default, 
                           "render_error": self.config.render_error_by_default}
        elif render_args is None:
            render_args = {}

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, render_args=render_args)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, render_args: Dict = None) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, render_args=render_args)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

        if not self.training and self.config.render_error_by_default and "error" in outputs:
            outputs["rgb"] = outputs["error"].repeat_interleave(3, dim=-1)

        return outputs

class _GradientScaler(torch.autograd.Function):  # typing: ignore
    """
    Scale gradients by a constant factor.
    """

    @staticmethod
    def forward(ctx, value, scaling):
        ctx.save_for_backward(scaling)
        return value, scaling

    @staticmethod
    def backward(ctx, output_grad, grad_scaling):
        (scaling,) = ctx.saved_tensors
        return output_grad * scaling, grad_scaling

def scale_gradients_by_distance_squared(
    field_outputs: Dict[FieldHeadNames, torch.Tensor], ray_samples: RaySamples
) -> Dict[FieldHeadNames, torch.Tensor]:
    """
    Scale gradients by the ray distance to the pixel
    as suggested in `Radiance Field Gradient Scaling for Unbiased Near-Camera Training` paper

    Note: The scaling is applied on the interval of [0, 1] along the ray!

    Example:
        GradientLoss should be called right after obtaining the densities and colors from the field. ::
            >>> field_outputs = scale_gradient_by_distance_squared(field_outputs, ray_samples)
    """
    out = {}
    ray_dist = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    scaling = torch.square(ray_dist).clamp(0, 1)
    for key, value in field_outputs.items():
        out[key], _ = cast(Tuple[Tensor, Tensor], _GradientScaler.apply(value, scaling))
    return out
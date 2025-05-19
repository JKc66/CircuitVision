import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import warnings
from torchvision.transforms import Normalize, Resize, ToTensor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.sam2_base import SAM2Base
from peft import LoraConfig, get_peft_model, TaskType
from sam2.utils.misc import get_connected_components



# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")



class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

class MultiKernelRefinement(nn.Module):
    """
    Applies multiple convolutional kernels in parallel for refinement
    and combines their outputs.
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_sizes=[3, 5, 7, 9, 11], intermediate_channels=8):
        """
        Args:
            in_channels (int): Number of input channels (usually 1 for mask logits).
            out_channels (int): Number of final output channels (usually 1 for refined logits).
            kernel_sizes (list[int]): List of odd kernel sizes for parallel conv branches.
            intermediate_channels (int): Number of output channels for EACH parallel conv branch.
        """
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.intermediate_channels_per_branch = intermediate_channels

        # Create parallel convolutional branches
        self.conv_branches = nn.ModuleList()
        for k_size in kernel_sizes:
            # padding='same' ensures output H, W match input H, W for odd kernels
            # For even kernels, padding needs manual calculation: padding = (k_size - 1) // 2
            if k_size % 2 == 0:
                 raise ValueError(f"Even kernel size {k_size} not directly supported with padding='same'. Use odd kernels or calculate padding manually.")
            branch = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.intermediate_channels_per_branch,
                kernel_size=k_size,
                padding='same', # Works for odd kernel sizes
                bias=True
            )
            self.conv_branches.append(branch)

        # Activation function after each branch (optional but common)
        self.activation = nn.GELU() # Or nn.GELU() etc.

        # Final combination layer
        # Takes concatenated features from all branches
        total_intermediate_channels = len(kernel_sizes) * self.intermediate_channels_per_branch
        self.combiner_conv = nn.Conv2d(
            in_channels=total_intermediate_channels,
            out_channels=out_channels,
            kernel_size=1, # 1x1 convolution to combine features channel-wise
            padding=0,
            bias=True
        )

    def forward(self, x):
        branch_outputs = []
        for branch_conv in self.conv_branches:
            branch_out = self.activation(branch_conv(x)) # Apply conv then activation
            branch_outputs.append(branch_out)

        # Concatenate outputs along the channel dimension
        concatenated_features = torch.cat(branch_outputs, dim=1) # Shape: (B, total_intermediate_channels, H, W)

        # Combine features using the 1x1 convolution
        refined_output = self.combiner_conv(concatenated_features) # Shape: (B, out_channels, H, W)

        return refined_output

class SAM2ImageWrapper(nn.Module):
    """
    A wrapper around SAM2Base for image-only segmentation.
    Applies LoRA internally to the wrapped model.
    """
    def __init__(self, modified_sam2_model: SAM2Base, embedding_r=4, use_refinement=False, refinement_kernel_sizes=[3, 5, 7, 9, 11]):
        super().__init__()
        self.sam2_model = modified_sam2_model
        self.use_refinement = use_refinement
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        
        self.embedding_r = embedding_r
        self.dense_embedding1 = nn.Parameter(torch.randn(1, 256, self.embedding_r))
        self.dense_embedding2 = nn.Parameter(torch.randn(1, self.embedding_r, 64 * 64))
        self.sparse_embedding = nn.Parameter(torch.randn(1, 32, 256))
        if self.use_refinement:
            self.refinement_layer = MultiKernelRefinement(
                in_channels=1,
                out_channels=1,
                kernel_sizes=refinement_kernel_sizes, # Example kernel sizes
                intermediate_channels=4 # Example intermediate channels per branch
            )
        else:
            self.refinement_layer = None

    def forward(self, images, points=None, point_labels=None, masks_prompt=None, multimask_output=False):
        """
        Simplified forward pass using the wrapped SAM2Base's methods.
        """

        # 1. Encode Image
        out = self.sam2_model.image_encoder(images)
        out["backbone_fpn"][0] = self.sam2_model.sam_mask_decoder.conv_s0(
            out["backbone_fpn"][0]
        )
        out["backbone_fpn"][1] = self.sam2_model.sam_mask_decoder.conv_s1(
            out["backbone_fpn"][1]
        )

        # 2. Prepare Decoder Inputs
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(out)
        # --- Corrected List Comprehension ---
        feats = [
            # Get Batch Size (B) dynamically from the input feature tensor
            feat.permute(1, 2, 0).view(feat.shape[1], -1, *feat_size)
            #                            ^^^^^^^^^^^  Use B from feat.shape[1]
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1] # Reverse the resulting list
        # --- Dictionary Creation (remains the same) ---
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        # 4. Run SAM Prompt Encoder and Mask Decoder directly
        high_res_features = _features["high_res_feats"]

        # compute the trainable prompt embedding
        dense_embedding = (self.dense_embedding1 @ self.dense_embedding2).view(1, 256, 64, 64)
        
        low_res_masks, iou_predictions, _, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=_features["image_embed"],
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )

        # 5. Return desired outputs
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(self.sam2_model.image_size, self.sam2_model.image_size),
            mode="bilinear",
            align_corners=False,
        )


        if self.use_refinement:
            high_res_masks = self.refinement_layer(high_res_masks)
            
        
        return high_res_masks, low_res_masks, iou_predictions
    
def get_modified_sam2(
    # --- Model Config ---
    model_cfg_path: str,
    checkpoint_path: str,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    use_high_res_features: bool = True,
    # --- PEFT Config ---
    use_peft: bool = True,
    lora_rank: int = 12,
    lora_alpha: int = 16,
    lora_dropout: float = 0.2,
    lora_target_modules: list = None, # List of module names (strings)
    # --- Wrapper/Task Config ---
    use_wrapper: bool = True,
    trainable_embedding_r: int = 4,
    # --- Refinement Layer ---
    use_refinement_layer: bool = False,
    refinement_kernels: list = [3, 5, 7, 11],
    kernel_channels: int = 4,
    # --- Loss Settings ---
    weight_dice=0.5, weight_focal=0.4, weight_iou=0.1, 
    weight_tversky: float = 0.0, weight_tv: float = 0.0, weight_freq: float = 0.0,
    dice_smooth=1e-5, focal_alpha=0.25, focal_gamma=2.0,
    iou_smooth=1e-5, iou_threshold=0.5,
    tversky_alpha = 0.2, tversky_beta= 0.8,
    apply_sigmoid=True,
    # --- Optimizer Settings ---
    lr=1e-3
    ):
    """
    Initializes SAM 2, applies PEFT/LoRA, and optionally wraps it for image-only tasks.

    Args:
        model_cfg_path (str): Path to the SAM 2 model config YAML.
        checkpoint_path (str): Path to the SAM 2 model checkpoint (.pt file).
        device (str): Device to load the model onto ('cuda', 'cpu', 'mps').
        use_high_res_features (bool): Whether the decoder should use high-res skip connections.
        use_peft (bool): Whether to apply PEFT/LoRA.
        lora_rank (int): Rank for LoRA matrices.
        lora_alpha (int): Alpha scaling for LoRA.
        lora_dropout (float): Dropout probability for LoRA layers.
        lora_target_modules (list): List of specific module names (strings) within the original
                                     SAM2 model to apply LoRA to. If None, PEFT might guess or
                                     you might need to define defaults.
        use_wrapper (bool): Whether to wrap the (PEFT-)modified model in SAM2ImageWrapper.
        trainable_embedding_r (int): Rank factor for the trainable prompt embeddings in the wrapper.
        use_refinement_layer (bool): Whether to add the MultiKernelRefinement layer in the wrapper.

    Returns:
        torch.nn.Module: The potentially PEFT-modified and wrapped SAM 2 model.
    """
    print("--- Initializing Modified SAM 2 ---")
    model_device = torch.device(device)

    # 1. Load Original SAM 2 Model
    print(f"Loading SAM 2 from config: {model_cfg_path} and checkpoint: {checkpoint_path}")
    original_sam2_model = build_sam2(
        model_cfg_path,
        checkpoint_path,
        device=model_device,
        mode="train" # Keep in train mode for fine-tuning
    )
    original_sam2_model.use_high_res_features_in_sam = use_high_res_features
    print(f"Original model loaded on {model_device}. use_high_res_features_in_sam set to {use_high_res_features}.")

    # --- Model to be returned ---
    final_model = original_sam2_model

    # 2. Apply PEFT/LoRA if enabled
    if use_peft:
        print(f"Applying PEFT/LoRA with rank={lora_rank}, alpha={lora_alpha}")
        if lora_target_modules is None:
            # Define default target modules if none provided
            # These defaults should cover key areas for image-only fine-tuning
            lora_target_modules = [
                # Decoder Transformer Attention (Self and Cross)
                "sam_mask_decoder.transformer.layers.0.self_attn.k_proj",
            ]
            print(f"Using default lora_target_modules: {lora_target_modules}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none", # Common setting
            modules_to_save=None, # Only train LoRA parameters
            init_lora_weights=True, # Default initialization
        )

        # Apply PEFT
        # Note: get_peft_model freezes non-target layers automatically
        peft_model = get_peft_model(original_sam2_model, lora_config)
        print("PEFT LoRA configuration applied.")
        peft_model.print_trainable_parameters()
        final_model = peft_model # Update the model to be returned
    else:
        print("Skipping PEFT/LoRA application. Freezing all parameters.")
        # Freeze all parameters if not using PEFT, as wrapper adds new ones
        for param in final_model.parameters():
             param.requires_grad = False
            
    # 3. Apply Wrapper if enabled
    if use_wrapper:
        print("Applying SAM2ImageWrapper...")
        wrapped_model = SAM2ImageWrapper(
            modified_sam2_model=final_model, # Pass the (potentially PEFT-modified) model
            embedding_r=trainable_embedding_r,
            use_refinement=use_refinement_layer,
            refinement_kernel_sizes=refinement_kernels
        )
        final_model = wrapped_model.to(model_device) # Update the model to be returned
        print("Wrapper applied.")
    else:
        print("Skipping SAM2ImageWrapper.")

    # 4. Final Verification of Trainable Parameters
    print("\n--- Final Trainable Parameters ---")
    total_trainable = 0
    for name, param in final_model.named_parameters():
        if param.requires_grad:
            print(f"- {name}: {param.shape} ({param.numel()})")
            total_trainable += param.numel()
    print(f"Total Trainable Parameters in Final Model: {total_trainable}")
    if total_trainable == 0 and (use_peft or use_wrapper):
         print("WARNING: No trainable parameters found! Check PEFT config and wrapper parameter initialization.")
    elif not use_peft and not use_wrapper and total_trainable > 0:
         print("Warning: Model has trainable parameters but PEFT/Wrapper were not used?")

    print("--- Modified SAM2 Structure Ready ---")
    return final_model


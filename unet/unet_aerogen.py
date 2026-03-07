"""
AeroGen UNet: diffusers ModelMixin wrapper for the custom bbox-conditioned UNet.

Self-contained - no ldm/bldm dependency. Uses local openaimodel_bbox.UNetModel.
"""

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .openaimodel_bbox import UNetModel


class AeroGenUNet2DConditionModel(ModelMixin, ConfigMixin):
    """
    Diffusers-compatible wrapper for AeroGen's bbox-conditioned UNet.
    Forward signature: x, timesteps, context, control, category_control, mask_control.
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        attention_resolutions: tuple = (4, 2, 1),
        num_res_blocks: int = 2,
        channel_mult: tuple = (1, 2, 4, 4),
        num_heads: int = 8,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: int = 768,
        use_checkpoint: bool = True,
        legacy: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=list(attention_resolutions),
            channel_mult=list(channel_mult),
            num_heads=num_heads,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            use_checkpoint=use_checkpoint,
            legacy=legacy,
            **kwargs,
        )

    def forward(
        self,
        x,
        timesteps,
        context=None,
        control=None,
        category_control=None,
        mask_control=None,
        **kwargs,
    ):
        return self.model(
            x,
            timesteps,
            context=context,
            control=control,
            category_control=category_control or [],
            mask_control=mask_control or [],
            **kwargs,
        )

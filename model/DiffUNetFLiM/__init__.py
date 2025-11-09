import torch
import torch.nn as nn
from typing import Sequence
import math
from monai.networks.nets import DiffusionModelUNet

class FiLMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation: out = gamma * x + beta"""

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) feature maps
            gamma: (B, C) scale parameters
            beta: (B, C) shift parameters
        Returns:
            modulated features (B, C, D, H, W)
        """
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1, 1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1, 1)

        return gamma * x + beta


class FiLMAdapter(nn.Module):
    """
    Generates FiLM parameters (gamma, beta) from volume and spacing information.
    Outputs modulation parameters for each U-Net resolution level.
    """

    def __init__(
        self,
        unet_channels: Sequence[int] = (32, 64, 64, 64),
        embed_dim: int = 256,
        volume_mean: float = 150.0,
        volume_std: float = 100.0,
        use_log_volume: bool = False,
    ):
        """
        Args:
            unet_channels: Channel dimensions at each U-Net level (should match your U-Net)
            embed_dim: Dimension of the intermediate embedding
            volume_mean: Mean volume for normalization (compute from your dataset)
            volume_std: Std volume for normalization (compute from your dataset)
            use_log_volume: If True, use log normalization instead of standard normalization
        """
        super().__init__()
        self.unet_channels = list(unet_channels)
        self.volume_mean = volume_mean
        self.volume_std = volume_std
        self.use_log_volume = use_log_volume

        # Base embedding network for volume + spacing
        self.embedding_net = nn.Sequential(
            nn.Linear(4, 64),  # 4 = volume (1) + spacing (3)
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, embed_dim),
            nn.SiLU(),
        )

        # FiLM parameter generators for each U-Net level
        # Each generates both gamma and beta (2 * channels)
        self.film_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, 2 * channels),
                )
                for channels in self.unet_channels
            ]
        )

        # Learnable scale factors for each level (helps with initialization)
        self.gamma_scales = nn.ParameterList(
            [
                nn.Parameter(torch.ones(channels) * 0.1)
                for channels in self.unet_channels
            ]
        )
        self.beta_scales = nn.ParameterList(
            [
                nn.Parameter(torch.ones(channels) * 0.1)
                for channels in self.unet_channels
            ]
        )

    def normalize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Normalize volume to a reasonable range"""
        if self.use_log_volume:
            # Log normalization (better for volumes with large variance)
            volume = torch.log(volume + 1.0)
            log_mean = torch.log(torch.tensor(self.volume_mean + 1.0))
            log_std = torch.log(torch.tensor(self.volume_std + 1.0))
            return (volume - log_mean) / log_std
        else:
            # Standard normalization
            return (volume - self.volume_mean) / self.volume_std

    def forward(
        self, volume: torch.Tensor, spacing: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            volume: (B,) or (B, 1) - organ volumes in ml
            spacing: (B, 3) - [spacing_x, spacing_y, spacing_z] in mm

        Returns:
            List of (gamma, beta) tuples, one for each U-Net level
            Each gamma and beta has shape (B, C) where C is channels at that level
        """
        # Normalize volume
        if volume.dim() == 1:
            volume = volume.unsqueeze(-1)  # (B, 1)

        volume_normalized = self.normalize_volume(volume)

        # Concatenate volume and spacing
        volume_spacing = torch.cat([volume_normalized, spacing], dim=-1)  # (B, 4)

        # Generate base embedding
        base_embed = self.embedding_net(volume_spacing)  # (B, embed_dim)

        # Generate FiLM parameters for each level
        film_params = []
        for i, (film_gen, gamma_scale, beta_scale) in enumerate(
            zip(self.film_generators, self.gamma_scales, self.beta_scales)
        ):
            params = film_gen(base_embed)  # (B, 2*C)
            gamma_raw, beta_raw = params.chunk(2, dim=-1)  # Each (B, C)

            # Scale and shift to initialize near identity transform
            # gamma starts near 1.0, beta starts near 0.0
            gamma = gamma_raw * gamma_scale + 1.0
            beta = beta_raw * beta_scale

            film_params.append((gamma, beta))

        return film_params


class DiffusionModelUNetFiLM(DiffusionModelUNet):
    """
    DiffusionModelUNet with FiLM (Feature-wise Linear Modulation) conditioning.
    Applies FiLM after each down block, middle block, and up block.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FiLM layers for each resolution level
        # down_blocks + middle + up_blocks
        num_levels = len(self.block_out_channels)

        # FiLM for down blocks
        self.film_down = nn.ModuleList([FiLMLayer() for _ in range(num_levels)])

        # FiLM for middle block
        self.film_mid = FiLMLayer()

        # FiLM for up blocks
        self.film_up = nn.ModuleList([FiLMLayer() for _ in range(num_levels)])

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims)
            timesteps: timestep tensor (N,)
            film_params: List of (gamma, beta) tuples for FiLM conditioning.
                        Should have length = len(channels) for down blocks.
                        If None, no FiLM conditioning is applied.
            context: context tensor for cross-attention (N, 1, ContextDim)
            class_labels: class labels (N,)
            down_block_additional_residuals: additional residuals for controlnet
            mid_block_additional_residual: additional residual for controlnet
        """
        # 1. Time embedding
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. Class embedding
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. Initial convolution
        h = self.conv_in(x)

        # 4. Down blocks with FiLM
        if context is not None and self.with_conditioning is False:
            raise ValueError(
                "model should have with_conditioning = True if context is provided"
            )

        down_block_res_samples: list[torch.Tensor] = [h]
        for i, downsample_block in enumerate(self.down_blocks):
            h, res_samples = downsample_block(
                hidden_states=h, temb=emb, context=context
            )

            # Apply FiLM modulation after the down block
            if film_params is not None and i < len(film_params):
                gamma, beta = film_params[i]
                h = self.film_down[i](h, gamma, beta)

                # Also apply FiLM to residual connections
                res_samples = [
                    self.film_down[i](res, gamma, beta) for res in res_samples
                ]

            for residual in res_samples:
                down_block_res_samples.append(residual)

        # Additional residuals for ControlNet
        if down_block_additional_residuals is not None:
            new_down_block_res_samples: list[torch.Tensor] = []
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples.append(down_block_res_sample)
            down_block_res_samples = new_down_block_res_samples

        # 5. Middle block with FiLM
        h = self.middle_block(hidden_states=h, temb=emb, context=context)

        if film_params is not None:
            # Use the last down block's FiLM params for middle block
            gamma, beta = film_params[-1]
            h = self.film_mid(h, gamma, beta)

        # Additional residual for ControlNet
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        # 6. Up blocks with FiLM
        for i, upsample_block in enumerate(self.up_blocks):
            idx: int = -len(upsample_block.resnets)  # type: ignore
            res_samples = down_block_res_samples[idx:]
            down_block_res_samples = down_block_res_samples[:idx]

            h = upsample_block(
                hidden_states=h,
                res_hidden_states_list=res_samples,
                temb=emb,
                context=context,
            )

            # Apply FiLM modulation after up block
            # Use corresponding down block's parameters (mirrored)
            if film_params is not None:
                film_idx = len(self.up_blocks) - 1 - i  # Mirror the indices
                if film_idx < len(film_params):
                    gamma, beta = film_params[film_idx]
                    h = self.film_up[i](h, gamma, beta)

        # 7. Output block
        output: torch.Tensor = self.out(h)

        return output


def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding

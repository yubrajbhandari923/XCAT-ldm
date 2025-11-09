import torch
from functools import partial
import tqdm
from monai.networks.schedulers import RFlowScheduler, Scheduler


class Sampler:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
    
    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model,
        scheduler = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
        cfg: float | None = None,
        cfg_fill_value: float = -1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
            cfg: classifier-free-guidance scale, which indicates the level of strengthening on the conditioning.
            cfg_fill_value: the fill value to use for the unconditioned input when using classifier-free guidance.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if mode == "concat" and conditioning is None:
            raise ValueError("Conditioning must be supplied for if condition mode is concat.")
        if not scheduler:
            scheduler = self.scheduler
        image = input_noise

        all_next_timesteps = torch.cat((scheduler.timesteps[1:], torch.tensor([0], dtype=scheduler.timesteps.dtype)))
        if verbose :
            progress_bar = tqdm(
                zip(scheduler.timesteps, all_next_timesteps),
                total=min(len(scheduler.timesteps), len(all_next_timesteps)),
            )
        else:
            progress_bar = iter(zip(scheduler.timesteps, all_next_timesteps))
        intermediates = []

        for t, next_t in progress_bar:
            # 1. predict noise model_output
            if (
                cfg is not None
            ):  # if classifier-free guidance is used, a conditioned and unconditioned bit is generated.
                model_input = torch.cat([image] * 2, dim=0)
                if conditioning is not None:
                    uncondition = torch.ones_like(conditioning)
                    uncondition.fill_(cfg_fill_value)
                    conditioning_input = torch.cat([uncondition, conditioning], dim=0)
                else:
                    conditioning_input = None
            else:
                model_input = image
                conditioning_input = conditioning
            if mode == "concat" and conditioning_input is not None:
                model_input = torch.cat([model_input, conditioning_input], dim=1)
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), context=None
                )
            else:
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning_input
                )
            if cfg is not None:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + cfg * (model_output_cond - model_output_uncond)

            # 2. compute previous image: x_t -> x_t-1
            if not isinstance(scheduler, RFlowScheduler):
                image, _ = scheduler.step(model_output, t, image)  # type: ignore
            else:
                image, _ = scheduler.step(model_output, t, image, next_t)  # type: ignore
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)

        if save_intermediates:
            return image, intermediates
        else:
            return image

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import (FlowMatchEulerDiscreteScheduler,
                       FlowMatchHeunDiscreteScheduler)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class FlowMatchHeunDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteBackwardScheduler(FlowMatchEulerDiscreteScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting=False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        margin_index_from_noise: int = 3,
        margin_index_from_image: int = 1,
        intermediate_steps=None
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
        )
        self.margin_index_from_noise = margin_index_from_noise
        self.margin_index_from_image = margin_index_from_image
        self.intermediate_steps = intermediate_steps

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps

        if num_inference_steps is None:
            num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = torch.cat([timesteps, torch.zeros(1, device=timesteps.device)])
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = self.timesteps.flip(0)
        self.sigmas = self.sigmas.flip(0)

        self.timesteps = self.timesteps[
            self.config.margin_index_from_image : num_inference_steps - self.config.margin_index_from_noise
        ]
        self.sigmas = self.sigmas[
            self.config.margin_index_from_image : num_inference_steps - self.config.margin_index_from_noise + 1
        ]

        if self.config.intermediate_steps is not None:
            # self.timesteps = torch.linspace(self.timesteps[0], self.timesteps[-1], self.config.intermediate_steps).to(self.timesteps.device)
            self.sigmas = torch.linspace(self.sigmas[0], self.sigmas[-1], self.config.intermediate_steps + 1).to(self.timesteps.device)
            self.timesteps = self.sigmas[:-1] * 1000


        self._step_index = None
        self._begin_index = None


class FlowMatchEulerDiscreteForwardScheduler(FlowMatchEulerDiscreteScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting=False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        margin_index_from_noise: int = 3,
        margin_index_from_image: int = 0,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
        )
        self.margin_index_from_noise = margin_index_from_noise
        self.margin_index_from_image = margin_index_from_image

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps

        if num_inference_steps is None:
            num_inference_steps = len(sigmas)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = self.timesteps[
            self.config.margin_index_from_noise : num_inference_steps - self.config.margin_index_from_image
        ]
        self.sigmas = self.sigmas[
            self.config.margin_index_from_noise : num_inference_steps - self.config.margin_index_from_image + 1
        ]

        self._step_index = None
        self._begin_index = None



class FlowMatchHeunDiscreteForwardScheduler(FlowMatchHeunDiscreteScheduler):
    _compatibles = []
    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        margin_index: int = 0,
        use_dynamic_shifting = False
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        
        self.use_dynamic_shifting = use_dynamic_shifting
        self.margin_index = margin_index

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self, 
        num_inference_steps: int = None, 
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """


        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )
            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.config.num_train_timesteps
        timesteps = timesteps[self.config.margin_index:]
        timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
        self.timesteps = timesteps.to(device=device)

        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        sigmas = sigmas[self.config.margin_index:]
        self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])

        # empty dt and derivative
        self.prev_derivative = None
        self.dt = None

        self._step_index = None
        self._begin_index = None

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchHeunDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_Heun_discrete.HeunDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_Heun_discrete.HeunDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_Heun_discrete.HeunDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `HeunDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        
        if self.state_in_first_order:
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
        else:
            # 2nd order / Heun's method
            sigma = self.sigmas[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        if self.state_in_first_order:
            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            denoised = sample - model_output * sigma
            # 2. convert to an ODE derivative for 1st order
            derivative = (sample - denoised) / sigma_hat
            # 3. Delta timestep
            dt = sigma_next - sigma_hat

            # store for 2nd order step
            self.prev_derivative = derivative
            self.dt = dt
            self.sample = sample
        else:
            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            denoised = sample - model_output * sigma_next
            # 2. 2nd order / Heun's method
            derivative = (sample - denoised) / sigma_next
            derivative = 0.5 * (self.prev_derivative + derivative)

            # 3. take prev timestep & sample
            dt = self.dt
            sample = self.sample

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.prev_derivative = None
            self.dt = None
            self.sample = None

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return prev_sample


class FlowMatchHeunDiscreteBackwardScheduler(FlowMatchHeunDiscreteScheduler):
    _compatibles = []
    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        margin_index: int = 0,
        use_dynamic_shifting = False
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        
        self.use_dynamic_shifting = use_dynamic_shifting
        self.margin_index = margin_index

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self, 
        num_inference_steps: int = None, 
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """


        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )
            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.config.num_train_timesteps
        timesteps = timesteps[self.config.margin_index:].flip(0)
        timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
        self.timesteps = timesteps.to(device=device)

        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        sigmas = sigmas[self.config.margin_index:].flip(0)
        self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])

        # empty dt and derivative
        self.prev_derivative = None
        self.dt = None

        self._step_index = None
        self._begin_index = None


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchHeunDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_Heun_discrete.HeunDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_Heun_discrete.HeunDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_Heun_discrete.HeunDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `HeunDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        
        if self.state_in_first_order:
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
        else:
            # 2nd order / Heun's method
            sigma = self.sigmas[self.step_index - 1]
            sigma_next = self.sigmas[self.step_index]
        
        if sigma == 0:
            prev_sample = sample + (sigma_next - sigma) * model_output
            prev_sample = prev_sample.to(model_output.dtype)

            # upon completion increase step index by one
            self._step_index += 2

            if not return_dict:
                return (prev_sample,)

            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        if self.state_in_first_order:
            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            denoised = sample - model_output * sigma
            # 2. convert to an ODE derivative for 1st order
            derivative = (sample - denoised) / sigma_hat
            # 3. Delta timestep
            dt = sigma_next - sigma_hat

            # store for 2nd order step
            self.prev_derivative = derivative
            self.dt = dt
            self.sample = sample
        else:
            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            denoised = sample - model_output * sigma_next
            # 2. 2nd order / Heun's method
            derivative = (sample - denoised) / sigma_next
            derivative = 0.5 * (self.prev_derivative + derivative)

            # 3. take prev timestep & sample
            dt = self.dt
            sample = self.sample

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.prev_derivative = None
            self.dt = None
            self.sample = None

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
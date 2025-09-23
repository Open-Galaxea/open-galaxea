from __future__ import annotations
from PIL import Image
import math
import logging
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any

import time
import functools

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from transformers import AutoImageProcessor
from diffusers.optimization import SchedulerType, Optimizer, TYPE_TO_SCHEDULER_FUNCTION

logger = logging.getLogger(__name__)


class ImageProcessorToTransform:
    def __init__(self, processor: AutoImageProcessor):
        """
        Initialize the wrapper with an ImageProcessor instance.

        Args:
            processor (AutoImageProcessor): The Hugging Face ImageProcessor instance
        """
        self.processor = processor

    def __call__(self, img: Image, **kwargs: str) -> torch.Tensor:
        """
        Process the input image and return a PyTorch tensor.
        
        Args:
            img (PIL.Image): The input image to process.
        
        Returns:
            torch.Tensor: Processed image as a tensor ready for model input.
        """
        # Process the image using the ImageProcessor
        inputs = self.processor(img, return_tensors="pt", **kwargs)
        
        # Return the 'pixel_values' which is the processed tensor
        return inputs['pixel_values']


class WarmUpCosineLRByIteration(LRScheduler):
    def __init__(self, optimizer, warmup_iterations, total_iterations, base_lr, min_lr=0, last_epoch=-1):
        self.warmup_iterations = warmup_iterations
        self.total_iterations = total_iterations
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.iteration = 0
        super(WarmUpCosineLRByIteration, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # If in the warmuop phase, linearly increase the learning rate
        if self.iteration < self.warmup_iterations:
            warmup_factor = (self.iteration + 1) / self.warmup_iterations
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        iteration_tensor = self.iteration - self.warmup_iterations
        total_iterations_tensor = self.total_iterations - self.warmup_iterations

        cosine_factor = 0.5 * (1 + math.cos(torch.pi * iteration_tensor / total_iterations_tensor))

        return [base_lr*cosine_factor + self.min_lr * (1 - cosine_factor) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        self.iteration += 1        
        super().step(epoch)


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)

def log_execution_time(logger=None):
    """Decorator to log the execution time of a function"""
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if logger is None:
                print(f"{func.__name__} took {elapsed_time:.2f} seconds to execute.")
            else:
                logger.info(
                    f"{func.__name__} took {elapsed_time:.2f} seconds to execute."
                )
            return result
            
        return wrapper
    
    return decorator
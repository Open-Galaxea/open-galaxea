import functools
import time

import torch
from contextlib import contextmanager

def log_allocated_gpu_memory(log=None, stage="loading model", device=0):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device)
        msg = f"Allocated GPU memory after {stage}: {allocated_memory/1024/1024/1024:.2f} GB"
        print(msg) if log is None else log.info(msg)
        return allocated_memory/1024/1024/1024


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


class Timer:
    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff

@contextmanager
def log_execution_time_ctx(enabled=True, logger=None):
    if not enabled:
        yield
        return

    start_time = time.time()
    try:
        yield  
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        if logger is None:
            print(f"Execution Time: {execution_time:.4f} seconds")
        else:
            logger.info(f"Execution Time: {execution_time:.4f} seconds")
            
@contextmanager
def track_time_gpu(name="Operation", disable=False):
    """
    Context manager to track the time of an operation.
    Args:
        name (str): The name of the operation (e.g., 'Forward Pass', 'Backward Pass')
    """
    if not disable:
        torch.cuda.synchronize() 
        start_time = time.time()

    try:
        # Yield control back to the block of code inside the 'with' statement
        yield
    finally:
        if not disable:
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            print(f"{name} took {elapsed_time:.4f} seconds")

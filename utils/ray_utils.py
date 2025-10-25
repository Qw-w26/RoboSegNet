"""Ray initialization and management utilities."""

import os
import warnings
from typing import Optional, Dict, Any
from pathlib import Path


def get_ray_context() -> Dict[str, Any]:
    """
    Get current Ray context information.

    Returns:
        Dictionary with Ray context info, or empty dict if Ray not initialized
    """
    try:
        import ray
        if ray.is_initialized():
            return {
                'initialized': True,
                'address': ray.get_runtime_context().get_session_dir(),
                'num_cpus': ray.cluster_resources().get('CPU', 0),
                'num_gpus': ray.cluster_resources().get('GPU', 0),
            }
        else:
            return {'initialized': False}
    except ImportError:
        return {'initialized': False, 'error': 'Ray not installed'}


def initialize_ray(
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    object_store_memory: Optional[int] = None,
    log_to_driver: bool = False,
    ignore_reinit_error: bool = True,
    include_dashboard: bool = False,
    **kwargs
) -> bool:
    """
    Initialize Ray with proper error handling and configuration.

    Args:
        num_cpus: Number of CPUs to use (None = all available)
        num_gpus: Number of GPUs to use (None = all available)
        object_store_memory: Object store memory in bytes (None = auto)
        log_to_driver: Whether to log to driver
        ignore_reinit_error: Whether to ignore reinitialization errors
        include_dashboard: Whether to include Ray dashboard
        **kwargs: Additional arguments to pass to ray.init()

    Returns:
        True if Ray was initialized, False if already initialized

    Example:
        >>> initialize_ray(num_gpus=4, object_store_memory=10*1024**3)
        >>> # Ray now initialized with 4 GPUs and 10GB object store
    """
    try:
        import ray
    except ImportError:
        raise ImportError(
            "Ray is not installed. Install with: pip install ray"
        )

    # Check if already initialized
    if ray.is_initialized():
        if ignore_reinit_error:
            warnings.warn("Ray is already initialized. Skipping initialization.")
            return False
        else:
            raise RuntimeError(
                "Ray is already initialized. Call shutdown_ray() first or "
                "set ignore_reinit_error=True"
            )

    # Build init kwargs
    init_kwargs = {
        'ignore_reinit_error': ignore_reinit_error,
        'log_to_driver': log_to_driver,
        'include_dashboard': include_dashboard,
    }

    if num_cpus is not None:
        init_kwargs['num_cpus'] = num_cpus

    if num_gpus is not None:
        init_kwargs['num_gpus'] = num_gpus

    if object_store_memory is not None:
        init_kwargs['object_store_memory'] = object_store_memory

    # Add any additional kwargs
    init_kwargs.update(kwargs)

    # Initialize Ray
    try:
        ray.init(**init_kwargs)
        print(f"Ray initialized successfully")
        print(f"  CPUs: {ray.cluster_resources().get('CPU', 0)}")
        print(f"  GPUs: {ray.cluster_resources().get('GPU', 0)}")
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Ray: {e}")


def shutdown_ray(verbose: bool = True):
    """
    Shutdown Ray and cleanup resources.

    Args:
        verbose: Whether to print shutdown message

    Example:
        >>> shutdown_ray()
        >>> # Ray is now shut down and resources are freed
    """
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            if verbose:
                print("Ray shutdown successfully")
        else:
            if verbose:
                print("Ray is not initialized, nothing to shutdown")
    except ImportError:
        if verbose:
            print("Ray is not installed")


def ensure_ray_initialized(
    num_gpus: Optional[int] = None,
    **kwargs
) -> None:
    """
    Ensure Ray is initialized, initializing if necessary.

    This is a convenience function that checks if Ray is initialized
    and initializes it if not.

    Args:
        num_gpus: Number of GPUs to use if initializing
        **kwargs: Additional arguments to pass to initialize_ray()

    Example:
        >>> ensure_ray_initialized(num_gpus=4)
        >>> # Ray is now guaranteed to be initialized
    """
    try:
        import ray
        if not ray.is_initialized():
            initialize_ray(num_gpus=num_gpus, **kwargs)
    except ImportError:
        raise ImportError(
            "Ray is not installed. Install with: pip install ray"
        )


class RayContext:
    """
    Context manager for Ray initialization and cleanup.

    Example:
        >>> with RayContext(num_gpus=4) as ray_ctx:
        ...     # Ray is initialized here
        ...     actor = ray.remote(MyActor).remote()
        ...     # Do work with Ray
        >>> # Ray is automatically shut down when exiting context
    """

    def __init__(
        self,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        shutdown_on_exit: bool = True,
        **kwargs
    ):
        """
        Initialize context manager.

        Args:
            num_cpus: Number of CPUs to use
            num_gpus: Number of GPUs to use
            shutdown_on_exit: Whether to shutdown Ray when exiting context
            **kwargs: Additional arguments to pass to initialize_ray()
        """
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.shutdown_on_exit = shutdown_on_exit
        self.kwargs = kwargs
        self.was_initialized = False

    def __enter__(self):
        """Enter context and initialize Ray if needed."""
        try:
            import ray
            self.was_initialized = ray.is_initialized()
        except ImportError:
            self.was_initialized = False

        if not self.was_initialized:
            initialize_ray(
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
                **self.kwargs
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and shutdown Ray if we initialized it."""
        if self.shutdown_on_exit and not self.was_initialized:
            shutdown_ray(verbose=False)

import os
import logging

logger = logging.getLogger(__name__)


def _parse_device_spec_for_worker(device_spec: str, worker_id: int) -> str:
    """
    Parse device specification string and return devices for this worker.
    
    Format: "(0,1,2),(3,4),(5,6,7)" where each group is for a worker rank.
    Worker 0 gets (0,1,2), worker 1 gets (3,4), etc.
    
    Args:
        device_spec: Device allocation string
        worker_id: Current worker rank (0-indexed)
    
    Returns:
        Comma-separated device IDs for this worker (e.g., "0,1,2")
    """
    # Remove outer spaces and split by ")(" to get device groups
    cleaned = device_spec.strip().replace(" ", "")
    groups = cleaned.split("),(")
    
    if worker_id >= len(groups):
        raise ValueError(
            f"Worker {worker_id} exceeds device spec groups. "
            f"Device spec has {len(groups)} groups but worker_id={worker_id}"
        )
    
    # Get the group for this worker and clean up parentheses
    group = groups[worker_id]
    # Remove leading "(" from first group and trailing ")" from last group
    return group.lstrip("(").rstrip(")")


def setup_cpu_threading_limits(cpu_threads: str, num_threads: int = 1):
    """Set up CPU threading limits for PyTorch to prevent CPU oversubscription"""
    import torch
    os.environ["OMP_NUM_THREADS"] = cpu_threads
    os.environ["MKL_NUM_THREADS"] = cpu_threads
    os.environ["TORCH_NUM_THREADS"] = cpu_threads
    if torch.get_num_threads() != num_threads:
        torch.set_num_threads(num_threads)
    if torch.get_num_interop_threads() != num_threads:
        torch.set_num_interop_threads(num_threads)


def setup_worker_environment(
    worker_id: str, cpu_threads: str = "2", num_threads: int = 1
):
    """Set up environment variables and configuration for a device worker"""
    setup_cpu_threading_limits(cpu_threads, num_threads)

    # Get device spec if provided, otherwise use worker_id directly
    device_spec = os.environ.get("TT_VISIBLE_DEVICES_SPEC")
    if device_spec:
        # Parse spec and get devices for this worker rank
        worker_rank = int(worker_id)
        visible_devices = _parse_device_spec_for_worker(device_spec, worker_rank)
        logger.info(f"[TT-Plugin] Worker {worker_id}: Parsed device spec '{device_spec}' -> devices '{visible_devices}'")
    else:
        # Fallback: no device restriction (use all available)
        visible_devices = None
        logger.info(f"[TT-Plugin] Worker {worker_id}: No TT_VISIBLE_DEVICES_SPEC set, using all devices")

    # Set device visibility only if we have a specific spec
    if visible_devices is not None:
        os.environ["TT_VISIBLE_DEVICES"] = visible_devices
        os.environ["TT_METAL_VISIBLE_DEVICES"] = visible_devices

    # Set per-worker cache directory if TT_METAL_HOME is set
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if tt_metal_home and visible_devices is not None:
        cache_suffix = visible_devices.replace(',', '_')
        os.environ["TT_METAL_CACHE"] = f"{tt_metal_home}/built/{cache_suffix}"
        logger.info(f"[TT-Plugin] Worker {worker_id}: TT_METAL_CACHE={os.environ['TT_METAL_CACHE']}")

    is_galaxy = os.environ.get("TT_METAL_IS_GALAXY", "0") == "1"
    if is_galaxy:
        _setup_galaxy_mesh_config(tt_metal_home)


def _setup_galaxy_mesh_config(tt_metal_home: str):
    """Configure mesh graph descriptors for Galaxy hardware"""
    os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"

    mesh_descriptors = {
        (1, 1): "n150_mesh_graph_descriptor.textproto",
        (2, 1): "n300_mesh_graph_descriptor.textproto",
        (2, 4): "t3k_mesh_graph_descriptor.textproto",
    }

    # Get mesh shape from environment variable (format: "rows,cols")
    mesh_shape_str = os.environ.get("TT_METAL_MESH_SHAPE")
    if mesh_shape_str:
        try:
            rows, cols = map(int, mesh_shape_str.split(","))
            mesh_shape = (rows, cols)
        except (ValueError, AttributeError):
            return  # Invalid format, skip mesh config
    else:
        return  # No mesh shape specified, skip mesh config

    descriptor = mesh_descriptors.get(mesh_shape)
    if descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = (
            f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/{descriptor}"
        )


def initialize_device_worker(
    worker_id: str, num_torch_threads: int = 1
):
    """Initialize worker environment (device visibility, mesh config, thread limits).
    
    Device initialization and model warmup are handled by tt_llm.py.
    This function only prepares environment variables per worker rank.
    """
    setup_worker_environment(worker_id, num_threads=num_torch_threads)
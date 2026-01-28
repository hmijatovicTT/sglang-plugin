import ast
import logging
import os
from abc import ABC
from typing import Optional

import torch
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class BaseMetalDeviceRunner(ABC):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        self.device_id = device_id
        self.logger = logger
        self.ttnn_device = None

        self.set_torch_thread_limits(num_torch_threads)

        if not os.getenv("HF_TOKEN", None) and not (
            os.getenv("HF_HOME", None) and any(os.scandir(os.getenv("HF_HOME")))
        ):
            self.logger.warning(
                "HF_TOKEN environment variable is not set and no cached models found in HF_HOME. Some models may not load properly."
            )

        self.is_tensor_parallel = False

    def get_pipeline_device_params(self):
        """Return device params including trace_region_size for tracing support."""
        return {
            "trace_region_size": 50000000,  # ~50MB for trace buffers
        }

    def set_device(self):
        if self.ttnn_device is None:
            # Setup worker environment (device visibility, mesh config, thread limits)
            # Extracts dp_rank from process title and configures TT_VISIBLE_DEVICES
            from sglang_tt_plugin.patching.patching_worker import patch_worker_setup
            patch_worker_setup()
            
            # Now open device - will see only the devices assigned to this worker
            self.ttnn_device = self._mesh_device()
        server_args = get_global_server_args()
        self.max_batch_size = server_args.max_running_requests or 32
        return self.ttnn_device

    def close_device(self):
        import ttnn
        try:
            self.logger.info(f"Device {self.device_id}: Closing mesh device...")
            if self.ttnn_device is not None:
                ttnn.close_mesh_device(self.ttnn_device)
                self.logger.info(
                    f"Device {self.device_id}: Successfully closed mesh device"
                )
            else:
                self.logger.info(
                    f"Device {self.device_id}: Device is None, no need to close"
                )
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to close device: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Device cleanup failed: {str(e)}"
            ) from e

    def get_updated_device_params(self, device_params):
        import ttnn
        if device_params is None:
            device_params = {}

        new_device_params = device_params.copy()

        dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
        dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
        fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

        if ttnn.device.is_blackhole():
            # Only when both fabric_config and fabric_tensix_config are set, we can use ROW dispatch, otherwise force to use COL dispatch
            fabric_config = new_device_params.get("fabric_config", None)
            if not (fabric_config and fabric_tensix_config):
                # When not both are set, force COL dispatch
                if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                    self.logger.warning(
                        "ROW dispatch requires both fabric and tensix config, using DispatchCoreAxis.COL instead."
                    )
                    dispatch_core_axis = ttnn.DispatchCoreAxis.COL
            elif fabric_config and fabric_tensix_config:
                self.logger.warning(
                    f"Blackhole with fabric_config and fabric_tensix_config enabled, using fabric_tensix_config={fabric_tensix_config}"
                )

        dispatch_core_config = ttnn.DispatchCoreConfig(
            dispatch_core_type, dispatch_core_axis, fabric_tensix_config
        )
        new_device_params["dispatch_core_config"] = dispatch_core_config

        return new_device_params

    def _mesh_device(self):
        import ttnn
        try:
            # Get available devices
            device_ids = ttnn.get_device_ids()
            if not device_ids:
                raise RuntimeError("No TTNN devices available")
            
            num_devices = len(device_ids)
            self.logger.info(
                f"Device {self.device_id}: Found {num_devices} available TTNN devices: {device_ids}"
            )

            # Get mesh shape from device IDs count
            mesh_shape = ttnn.MeshShape(1, num_devices)

            # Configure fabric BEFORE opening mesh device
            fabric_config = self._configure_fabric(num_devices)

            device_params = self.get_pipeline_device_params()
            updated_device_params = self.get_updated_device_params(device_params)
            mesh_device = self._initialize_mesh_device(
                mesh_shape, updated_device_params, fabric_config
            )

            self.logger.info(
                f"Device {self.device_id}: Successfully created multidevice with {mesh_device.get_num_devices()} devices"
            )
            return mesh_device
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Unexpected error during device initialization: {e}"
            )
            raise RuntimeError(
                f"Unexpected device initialization error: {str(e)}"
            ) from e

    def _configure_fabric(self, num_devices: int):
        """Configure fabric before opening mesh device."""
        import ttnn
        
        if num_devices == 1:
            self.logger.info(f"Device {self.device_id}: Single device, no fabric config needed")
            return None
        
        # Detect if we're on Galaxy (multi-chip system like T3K)
        is_galaxy = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        
        # Use FABRIC_1D_RING for Galaxy, FABRIC_1D otherwise
        fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if is_galaxy else ttnn.FabricConfig.FABRIC_1D
        
        self.logger.info(
            f"Device {self.device_id}: Setting fabric config to {fabric_config} "
            f"(is_galaxy={is_galaxy}, num_devices={num_devices})"
        )
        ttnn.set_fabric_config(fabric_config)
        
        return fabric_config

    def _initialize_mesh_device(self, mesh_shape, device_params, fabric_config):
        import ttnn
        try:
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
        except Exception as e:
            try:
                if fabric_config:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception as reset_error:
                self.logger.warning(
                    f"Device {self.device_id}: Failed to reset fabric after device initialization failure: {reset_error}"
                )
            self.logger.error(
                f"Device {self.device_id}: Mesh device initialization failed: {e}"
            )
            raise RuntimeError(f"Mesh device initialization failed: {str(e)}") from e
        return mesh_device
    
    def set_torch_thread_limits(self, num_threads: int = 1):
        if torch.get_num_threads() != num_threads:
            torch.set_num_threads(num_threads)
        if torch.get_num_interop_threads() != num_threads:
            torch.set_num_interop_threads(num_threads)
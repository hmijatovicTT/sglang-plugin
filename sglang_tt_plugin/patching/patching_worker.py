"""
Worker environment patching for TT-Metal device isolation.

This module extracts dp_rank from sglang's process title and calls
worker_setup to configure device visibility per worker.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)


def get_dp_rank_from_process_title() -> int:
    """Extract dp_rank from the current process title.
    
    sglang sets process title like "sglang::scheduler_DP1_TP0" in run_scheduler_process().
    Each forked worker process has its own title with its specific DP rank.
    
    Returns:
        int: DP rank extracted from title, or 0 if not found (dp=1 case)
    """
    import setproctitle
    
    title = setproctitle.getproctitle()
    match = re.search(r'_DP(\d+)', title)
    if match:
        dp_rank = int(match.group(1))
        logger.debug(f"[TT-Plugin] Extracted dp_rank={dp_rank} from process title: {title}")
        return dp_rank
    
    logger.debug(f"[TT-Plugin] No DP rank in process title '{title}', defaulting to 0")
    return 0


def patch_worker_setup():
    """Setup worker environment by extracting dp_rank from process title.
    
    This is called from BaseMetalDeviceRunner.set_device() before opening TT devices.
    Gets dp_rank from sglang's process title and delegates to worker_setup.
    """
    # Avoid running multiple times in same process
    if os.environ.get("_TT_WORKER_ENV_SETUP_DONE"):
        return
    
    dp_rank = get_dp_rank_from_process_title()
    
    from sglang_tt_plugin.worker_setup.worker_setup import setup_worker_environment
    setup_worker_environment(worker_id=str(dp_rank))
    
    os.environ["_TT_WORKER_ENV_SETUP_DONE"] = "1"
    logger.info(f"[TT-Plugin] Worker environment setup complete for dp_rank={dp_rank}")
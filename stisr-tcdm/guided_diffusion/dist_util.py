"""
Helpers for distributed training (patched for single-GPU without MPI).
"""

import io
import os
import socket
import torch as th
import torch.distributed as dist
import blobfile as bf

#Modified
"""
# Try to import MPI; if unavailable, skip MPI-related functionality
try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    MPI = None
    mpi_available = False
"""
mpi_available = False
MPI = None

# Change this to reflect your cluster layout.
GPUS_PER_NODE = 1
SETUP_RETRY_COUNT = 3


def setup_dist(gpu_id=0):
    """
    Setup a distributed process group.
    Skips MPI if mpi4py is not available (single-GPU mode).
    """
    if dist.is_initialized():
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if not mpi_available:
        # Single-GPU fallback: do nothing
        return

    # Original MPI setup
    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    Falls back to normal load if MPI is unavailable.
    """
    if not mpi_available:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)

    # Original MPI code
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    Skipped if MPI is unavailable.
    """
    if not mpi_available:
        return

    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
"""
*N:
Provides utilities for distributed training, modified to disable MPI and run safely on a single GPU by simplifying process group
setup, device handling, model loading, and parameter synchronization without multi-node communication.
"""
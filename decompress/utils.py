'''
/*
    * This file is part of DeepSpace.
    *
    * DeepSpace is free software: you can redistribute it and/or modify
    * it under the terms of the GNU Affero General Public License as published by
    * the Free Software Foundation, either version 3 of the License, or
    * (at your option) any later version.
    *
    * DeepSpace is distributed in the hope that it will be useful,
    * but WITHOUT ANY WARRANTY; without even the implied warranty of
    * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    * GNU Affero General Public License for more details.
    *
    * You should have received a copy of the GNU Affero General Public License
    * along with DeepSpace.  If not, see <https://www.gnu.org/licenses/>.
    */
'''
import os
import shutil

import torch
import torch.distributed as dist


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    # dist.broadcast requires CUDA backend (nccl); skip for single-process
    # or non-CUDA (MPS/CPU) runs where all ranks already share the same weights.
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    if not torch.cuda.is_available():
        return
    for param in params:
        dist.broadcast(param.data, src=0)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        backend = 'nccl'
    else:
        backend = 'gloo'  # gloo works on CPU and MPS (Mac)
    gpu = args.local_rank
    dist.init_process_group(
        backend=backend, init_method='env://', rank=rank, world_size=size)
    print(f"Rank {rank} initialized")
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()

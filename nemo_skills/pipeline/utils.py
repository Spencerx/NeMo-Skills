# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shlex
import subprocess
import sys
import tarfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import nemo_run as run
import yaml
from huggingface_hub import get_token
from invoke import StreamWatcher
from nemo_run.config import set_nemorun_home
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.execution.slurm import SlurmJobDetails, get_packaging_job_key
from nemo_run.core.tunnel import SSHTunnel
from omegaconf import DictConfig
from torchx.specs.api import AppState

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def wrap_arguments(arguments: str):
    """Returns a mock context object to allow using the cli entrypoints as functions."""

    class MockContext:
        def __init__(self, args):
            self.args = args
            self.obj = None

    # first one is the cli name
    return MockContext(args=arguments.split(" "))


# TODO: this file is way too big - we need to split it into pieces

# keeping a global variable for first submitted experiment (per cluster) and reusing it by default
# we are using ssh tunnel as a proxy for cluster identity, since even if other parameters are different
# we can still reuse code as long as ssh matches
REUSE_CODE_EXP = {}


@dataclass
class RepoMetadata:
    """Metadata for a repo that is used in the experiment."""

    name: str
    path: Path

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

        if not self.path.exists():
            raise ValueError(f"Repository path `{self.path}` does not exist.")


# Registry of external repos that should be packaged with the code in the experiment
EXTERNAL_REPOS = {
    'nemo_skills': RepoMetadata(
        name='nemo_skills', path=Path(__file__).absolute().parents[1]
    ),  # path to nemo_skills repo
}


def register_external_repo(metadata: RepoMetadata):
    """Register an external repo to be packaged with the code in the experiment.

    Args:
        metadata (RepoMetadata): Metadata for the external repo.
    """
    if metadata.name in EXTERNAL_REPOS:
        raise ValueError(f"External repo {metadata.name} is already registered.")

    EXTERNAL_REPOS[metadata.name] = metadata


def get_registered_external_repo(name: str) -> Optional[RepoMetadata]:
    """Get the path to the registered external repo.

    Args:
        name (str): Name of the external repo.

    Returns:
        A path to the external repo if it is registered, otherwise None.
    """
    if name not in EXTERNAL_REPOS:
        return None

    return EXTERNAL_REPOS[name]


def is_mounted_filepath(cluster_config: dict, path: str):
    """
    Check if the filepath is mounted using the cluster config. Does not raise an error if the filepath is not mounted.

    Args:
        cluster_config: cluster config dictionary
        path: path to the file to be mounted

    Returns:
        bool: Whether the filepath is mounted.
    """
    # Find which mount path matches the filepaths prefix
    for mount in get_mounts_from_config(cluster_config) + ['/nemo_run/code:/nemo_run/code']:
        mount_source, mount_dest = mount.split(':')
        if path.startswith(mount_dest):
            return True

    return False


def check_if_mounted(cluster_config, path_to_check):
    """Will check that path_to_check is referenced inside one of the mounts."""
    if cluster_config["executor"] == "none":
        # no mounts in local executor
        return

    if not is_mounted_filepath(cluster_config, path_to_check):
        raise ValueError(f"The path '{path_to_check}' is not mounted. Check cluster config.")


def check_mounts(
    cluster_config, log_dir: str, mount_map: Dict[str, Optional[str]] = None, check_mounted_paths: bool = False
):
    """
    Utility method to check if the paths are mounted, whether the remote directories exist, create them if they dont exist
    and finally resolve their mounted paths.

    Args:
        cluster_config: cluster config dictionary
        log_dir: optional str representing the log directory
        mount_map: a dictionary mapping the paths to their default mount locations.
            Keys must be the mount source, and values must be the mount destination.
            If the mount destination is None, the path will not be mounted but still checked and remote directory will still be created.
            Mount destinations must be absolute paths and begin with '/'.
        check_mounted_paths: if True, will perform remote calls to dynamically mount the provided paths, and create
            remote directories if required. Finally, will assert that all remote paths are valid files or directories.

    Returns:
        tuple: a tuple of the mounted paths for the provided paths in mount_map, and the log_dir
    """
    # Check if mount map is provided
    mount_map = mount_map or {}

    # Compute directory of all files
    remote_dir_list = []

    # Check paths and add to mount list if not mounted
    if check_mounted_paths:
        for idx, (path, default_mount) in enumerate(mount_map.items()):
            if not is_mounted_filepath(cluster_config, path):
                # check if the path is a file or a directory
                # so that the directory can be created
                is_file = os.path.splitext(path) != ""
                if is_file:
                    path_dir, _ = os.path.split(path)
                else:
                    path_dir = path
                remote_dir_list.append(path_dir)

                if default_mount is not None:
                    # Check that path is not empty and is an absolute path
                    assert (
                        default_mount[0] == '/'
                    ), f"Default mount path should be absolute path, given {default_mount}"

                    # Add mount path to the cluster config
                    add_mount_path(path_dir, default_mount, cluster_config)

    else:
        # Just check if the paths are mounted
        for path in mount_map.keys():
            check_if_mounted(cluster_config, path)

    # check if the paths are mounted, get them if they arent but have mount sources
    # will error out if there are no corresponding mount sources
    new_paths = [get_mounted_path(cluster_config, path) for path in mount_map.keys()]

    if log_dir:
        if check_mounted_paths:
            # Create log dir in some location that will be mounted
            remote_dir_list.append(log_dir)
        else:
            check_if_mounted(cluster_config, log_dir)
        log_dir = get_mounted_path(cluster_config, log_dir)

    if check_mounted_paths:
        # Create missing remote directories
        if remote_dir_list:
            create_remote_directory(remote_dir_list, cluster_config)

        # Check that the file or dir exists at the remote location
        check_remote_mount_directories(list(mount_map.keys()) + [log_dir], cluster_config)

    if new_paths:
        return *new_paths, log_dir
    return log_dir


def get_mounted_path(cluster_config: dict, path: str):
    """
    Resolve the mounted filepath using the cluster config to merge the mount destination path to the filepath.

    Args:
        cluster_config: cluster config dictionary
        path: path to the file to be mounted

    Returns:
        str: mounted filepath

    Raises:
        ValueError: if the filepath is not mounted
    """
    if cluster_config["executor"] == "none":
        # no mounts in local executor
        return path
    if path is None:
        return None

    # Find which mount path matches the filepaths prefix
    mount_path = None
    for mount in get_mounts_from_config(cluster_config) + ['/nemo_run/code:/nemo_run/code']:
        mount_source, mount_dest = mount.split(':')
        if path.startswith(mount_source):
            mount_path = mount
            break

        elif path.startswith(mount_dest):
            # already mounted, return immediately
            return path

    if mount_path is None:
        raise ValueError(
            f"Could not find a mount path for the file path `{path}`. Check cluster config. Below paths are mounted: \n"
            f"{cluster_config['mounts']}"
        )

    # replace the mount destination inside the filepath with the mount source
    mount_source, mount_dest = mount_path.split(':')
    # append the rest of the path to the mount destination
    filepath = mount_dest + path[len(mount_source) :]

    return filepath


def get_unmounted_path(cluster_config: dict, path: str):
    """
    Resolve the mounted filepath using the cluster config to merge the mount destination path to the filepath.
    If the filepath is already mounted, it will return the filepath as is.

    Args:
        cluster_config: cluster config dictionary
        path: path to the file to be mounted

    Returns:
        str: mounted filepath

    Raises:
        ValueError: if the filepath is not mounted
    """
    if cluster_config["executor"] == "none":
        # no mounts in local executor
        return path
    if path is None:
        return None

    # Find which mount path matches the filepaths prefix
    mount_path = None
    for mount in get_mounts_from_config(cluster_config) + ['/nemo_run/code:/nemo_run/code']:
        mount_source, mount_dest = mount.split(':')
        if path.startswith(mount_dest):
            mount_path = mount
            break

        elif path.startswith(mount_source):
            # already mounted, return immediately
            return path

    if mount_path is None:
        raise ValueError(
            f"Could not find a mount path for the file path `{path}`. Check cluster config. Below paths are mounted: \n"
            f"{cluster_config['mounts'] + ['/nemo_run/code:/nemo_run/code']}"
        )

    # replace the mount destination inside the filepath with the mount source
    mount_source, mount_dest = mount_path.split(':')

    # append the rest of the path to the mount source
    filepath = mount_source + path[len(mount_dest) :]  # replace the mount destination with the mount source

    return filepath


def add_mount_path(mount_source: str, mount_dest: str, cluster_config):
    """Add a mount path to the cluster configuration."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if 'mounts' in cluster_config:
        original_mounts = get_mounts_from_config(cluster_config) + ['/nemo_run/code:/nemo_run/code']
        added_mount = False
        for mount_path in original_mounts:
            source, destination = mount_path.split(':')

            if source == mount_source and destination == mount_dest:
                return

        if not added_mount:
            cluster_config['mounts'].append(f"{mount_source}:{mount_dest}")
            logging.info(f"Added mount path: `{mount_source}:{mount_dest}`")

    else:
        raise ValueError("No mounts found in cluster config, can only add to existing mount list.")


def create_remote_directory(directory: str | list, cluster_config: dict):
    """Create a remote directory on the cluster."""

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    if isinstance(directory, str):
        directory = [directory]

    # Get unmounted path of all directories
    directory = [
        get_unmounted_path(cluster_config, dir_path) if is_mounted_filepath(cluster_config, dir_path) else dir_path
        for dir_path in directory
    ]

    if cluster_config.get('executor') != 'slurm':
        tunnel = run.LocalTunnel(job_dir=directory[0])
        for dir_path in directory:
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} in local filesystem.")
        tunnel.cleanup()

    elif cluster_config.get('executor') == 'slurm':
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = directory[0]

        tunnel = get_tunnel(cluster_config)
        for dir_path in directory:
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} on remote cluster.")
        tunnel.cleanup()

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def resolve_mount_paths(cluster_config: dict, mount_paths: str | list | dict, create_remote_dir: bool = True):
    """
    Resolve the mount paths using the cluster config to merge the mount destination path to the filepath.
    Args:
        cluster_config: The cluster configuration dictionary to update.
        mount_paths: The mount paths to resolve - can be a string (comma separated), dict or a list of strings.
            Each mount path should be in the format `src:dest`.
        create_remote_dir: Whether to create the remote directories for the mount paths.
    """
    if mount_paths is not None:
        if isinstance(mount_paths, str):
            mount_paths_list = mount_paths.split(",")
        elif isinstance(mount_paths, dict):
            mount_paths_list = [f"{src}:{dest}" for src, dest in mount_paths.items()]
        else:
            mount_paths_list = mount_paths

        # remove empty strings from the list and strip whitespace
        mount_paths_list = [path.strip() for path in mount_paths_list]
        mount_paths_list = [path for path in mount_paths_list if path != ""]

        for idx, path in enumerate(mount_paths_list):
            assert ":" in path, f"Invalid mount path: {path}. Each path must be in the format `src:dest`"
            src, dest = path.split(":")
            src = src.strip().strip("\n")
            dest = dest.strip().strip("\n")

            LOG.info(f"Adding mount path:- {src}:{dest}")
            mount_paths_list[idx] = (src, dest)
            add_mount_path(src, dest, cluster_config)

        if create_remote_dir:
            LOG.info(f"Creating remote directories for mount paths:")
            all_src_dir = [src for src, _ in mount_paths_list]
            # Check if it is a file or a directory and only create the directory
            for idx in range(len(all_src_dir)):
                if os.path.splitext(all_src_dir[idx])[1] != "":
                    all_src_dir[idx] = os.path.dirname(all_src_dir[idx])
                LOG.info(f"Attempting to create remote directory: {all_src_dir[idx]}")

            create_remote_directory(all_src_dir, cluster_config)

    return cluster_config


def check_remote_mount_directories(directories: list, cluster_config: dict, exit_on_failure: bool = True):
    """Create a remote directory on the cluster."""
    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")
    if isinstance(directories, str):
        directories = [directories]

    # Get unmounted path of all directories
    directories = [
        get_unmounted_path(cluster_config, dir_path) if is_mounted_filepath(cluster_config, dir_path) else dir_path
        for dir_path in directories
    ]

    if cluster_config.get('executor') != 'slurm':
        tunnel = run.LocalTunnel(job_dir=None)
        all_dirs_exist = True
        missing_source_locations = []
        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)
            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)
        tunnel.cleanup()
        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )
    elif cluster_config.get('executor') == 'slurm':
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")
        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = os.getcwd()
        tunnel = get_tunnel(cluster_config)
        missing_source_locations = []
        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)
            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)
        tunnel.cleanup()
        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )
    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


# caching the status assuming it doesn't change while experiment is being scheduled
# otherwise this results in too many ssh calls
@lru_cache
def get_exp_handles(expname: str, ignore_finished=True, ignore_exp_not_exists=True) -> list[str]:
    """Will return the handles of the tasks in the experiment.

    If ignore_finished=True, will only return handles for the tasks
    that are not yet finished. Useful for filtering handles to set dependencies on.

    If ignore_exp_not_exists=True, will not raise an error if the experiment does not exist.

    TODO: it's still possible that job submission fails if the tasks exist when this function
          is called, but finish before nemo-run submits a new job (which might take minutes)
    """

    def _get_handles(exp):
        handles = []
        for job in exp.jobs:
            if not ignore_finished or (
                job.status(exp._runner) in [AppState.RUNNING, AppState.PENDING, AppState.SUBMITTED, AppState.UNKNOWN]
            ):
                handles.append(job.handle)
                continue
        return handles

    # if we are given an experiment object, we can directly get the handles
    if isinstance(expname, run.Experiment):
        return _get_handles(expname)

    try:
        with run.Experiment.from_title(expname) as exp:
            return _get_handles(exp)
    except FileNotFoundError:
        try:
            with run.Experiment.from_id(expname) as exp:
                return _get_handles(exp)
        except AssertionError:
            if ignore_exp_not_exists:
                LOG.warning("Experiment %s not found!", expname)
                return []
            raise ValueError(f"Experiment {expname} not found!")


def get_timeout(cluster_config, partition):
    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition or cluster_config["partition"]]

        # subtracting 15 minutes to account for the time it takes to save the model
        # the format expected by nemo is days:hours:minutes:seconds
        time_diff = datetime.strptime(timeout, "%H:%M:%S") - datetime.strptime("00:15:00", "%H:%M:%S")
        timeout = (
            f'00:{time_diff.seconds // 3600:02d}:{(time_diff.seconds % 3600) // 60:02d}:{time_diff.seconds % 60:02d}'
        )
    return timeout


def get_free_port(exclude: list[int] | None = None, strategy: int | str = 5000) -> int:
    """Will return a free port on the host."""
    exclude = exclude or []
    if isinstance(strategy, int):
        port = strategy
        while port in exclude:
            port += 1
        return port
    elif strategy == "random":
        import random

        port = random.randint(1024, 65535)
        while port in exclude:
            port = random.randint(1024, 65535)
        return port
    else:
        raise ValueError(f"Strategy {strategy} not supported.")


def get_generation_command(server_address, generation_commands):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        # might be required if we are not hosting server ourselves
        # this will try to handshake in a loop and unblock when the server responds
        f"echo 'Waiting for the server to start at {server_address}' && "
        f"while [ $(curl -X PUT {server_address} >/dev/null 2>&1; echo $?) -ne 0 ]; do sleep 3; done && "
        # will run in a single task always (no need to check mpi env vars)
        f"{generation_commands}"
    )
    return cmd


def get_reward_server_command(
    server_type: str,
    num_gpus: int,
    num_nodes: int,
    model_path: str,
    cluster_config: dict,
    server_port: int,
    server_args: str = "",
    server_entrypoint: str | None = None,
):
    num_tasks = num_gpus

    # check if the model path is mounted if not vllm;
    # vllm can also pass model name as "model_path" so we need special processing
    if server_type != "vllm":
        check_if_mounted(cluster_config, model_path)

    # the model path will be mounted, so generally it will start with /
    elif server_type == "vllm" and model_path.startswith("/"):
        check_if_mounted(cluster_config, model_path)

    if server_type == 'nemo':
        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_nemo_aligner_reward_model"
        nemo_aligner_reward_model_port = get_free_port(strategy="random", exclude=[server_port])
        server_start_cmd = (
            # Note: The order of the two commands is important as the reward model server
            # needs to be the first command so it can get the HF_TOKEN from the environment
            f"python -m {server_entrypoint} "
            f"    ++rm_model_file={model_path} "
            f"    trainer.devices={num_gpus} "
            f"    trainer.num_nodes={num_nodes} "
            f"    +model.tensor_model_parallel_size={num_gpus} "
            f"    +model.pipeline_model_parallel_size={num_nodes} "
            # This port could be configurable, but is hard coded to reduce
            # the divergence of the server command parameters from pipeline/generate.py
            f"    inference.port={nemo_aligner_reward_model_port} "
            f"    {server_args} & "
            f"python -m nemo_skills.inference.server.serve_nemo_reward_model "
            # These ports could be configurable, but is hard coded to reduce
            # the divergence of the server command parameters from pipeline/generate.py
            f"    inference_port={server_port}  "
            f"    triton_server_address=localhost:{nemo_aligner_reward_model_port} "
        )

        # somehow on slurm nemo needs multiple tasks, but locally only 1
        if cluster_config["executor"] != "slurm":
            num_tasks = 1

    elif server_type == "vllm":
        if num_nodes > 1:
            raise ValueError("VLLM server does not support multi-node execution")

        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_vllm"
        server_start_cmd = (
            f"python3 -m {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        num_tasks = 1
    else:
        raise ValueError(f"Server type '{server_type}' not supported for reward model.")

    server_cmd = (
        f"nvidia-smi && "
        f"cd /nemo_run/code && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"{server_start_cmd} "
    )
    return server_cmd, num_tasks


def get_ray_server_cmd(start_cmd):
    ports = (
        "--node-manager-port=12345 "
        "--object-manager-port=12346 "
        "--dashboard-port=8265 "
        "--dashboard-agent-grpc-port=12347 "
        "--runtime-env-agent-port=12349 "
        "--metrics-export-port=12350 "
        "--min-worker-port=14349 "
        "--max-worker-port=18349 "
    )

    ray_start_cmd = (
        "if [ \"${SLURM_PROCID:-0}\" = 0 ]; then "
        "    echo 'Starting head node' && "
        "    export RAY_raylet_start_wait_time_s=120 && "
        "    ray start "
        "        --head "
        "        --port=6379 "
        f"       {ports} && "
        f"   {start_cmd} ;"
        "else "
        "    echo 'Starting worker node' && "
        "    export RAY_raylet_start_wait_time_s=120 && "
        "    echo \"Connecting to head node at $SLURM_MASTER_NODE\" && "
        "    ray start "
        "        --block "
        "        --address=$SLURM_MASTER_NODE:6379 "
        f"       {ports} ;"
        "fi"
    )
    return ray_start_cmd


def get_server_command(
    server_type: str,
    num_gpus: int,
    num_nodes: int,
    model_path: str,
    cluster_config: dict,
    server_port: int,
    server_args: str = "",
    server_entrypoint: str | None = None,
):
    num_tasks = num_gpus

    # check if the model path is mounted if not vllm;
    # vllm can also pass model name as "model_path" so we need special processing
    if server_type != "vllm":
        check_if_mounted(cluster_config, model_path)

    # the model path will be mounted, so generally it will start with /
    elif server_type == "vllm" and model_path.startswith("/"):
        check_if_mounted(cluster_config, model_path)

    if server_type == 'nemo':
        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_nemo"
        server_start_cmd = (
            f"python -m {server_entrypoint} "
            f"    gpt_model_file={model_path} "
            f"    trainer.devices={num_gpus} "
            f"    trainer.num_nodes={num_nodes} "
            f"    tensor_model_parallel_size={num_gpus} "
            f"    pipeline_model_parallel_size={num_nodes} "
            f"    ++port={server_port} "
            f"    {server_args} "
        )

        # somehow on slurm nemo needs multiple tasks, but locally only 1
        if cluster_config["executor"] != "slurm":
            num_tasks = 1
    elif server_type == 'megatron':
        if cluster_config["executor"] != "slurm":
            num_tasks = 1
            prefix = f"torchrun --nproc_per_node {num_gpus}"
        else:
            prefix = "python "
        server_entrypoint = server_entrypoint or "tools/run_text_generation_server.py"
        # similar to conversion, we don't hold scripts for megatron on our side
        # and expect it to be in /opt/Megatron-LM in the container
        server_start_cmd = (
            f"export PYTHONPATH=$PYTHONPATH:/opt/Megatron-LM && "
            f"export CUDA_DEVICE_MAX_CONNECTIONS=1 && "
            f"cd /opt/Megatron-LM && "
            f"{prefix} {server_entrypoint} "
            f"    --load {model_path} "
            f"    --tensor-model-parallel-size {num_gpus} "
            f"    --pipeline-model-parallel-size {num_nodes} "
            f"    --use-checkpoint-args "
            f"    --max-tokens-to-oom 12000000 "
            f"    --micro-batch-size 1 "  # that's a training argument, ignored here, but required to specify..
            f"    {server_args} "
        )
    elif server_type == 'vllm':
        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_vllm"
        start_vllm_cmd = (
            f"python3 -m {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        server_start_cmd = get_ray_server_cmd(start_vllm_cmd)
        num_tasks = 1
    elif server_type == 'sglang':
        if num_nodes > 1:
            multinode_args = f" --dist_init_addr $SLURM_MASTER_NODE --node_rank $SLURM_PROCID "
        else:
            multinode_args = ""
        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_sglang"
        server_start_cmd = (
            f"python3 -m {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --num_nodes {num_nodes} "
            f"    --port {server_port} "
            f"    {multinode_args} "
            f"    {server_args} "
        )
        num_tasks = 1
    else:
        server_entrypoint = server_entrypoint or "nemo_skills.inference.server.serve_trt"
        # need this flag for stable Nemotron-4-340B deployment
        server_start_cmd = (
            f"FORCE_NCCL_ALL_REDUCE_STRATEGY=1 python -m {server_entrypoint} "
            f"    --model_path {model_path} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        num_tasks = num_gpus

    server_cmd = (
        f"nvidia-smi && "
        f"cd /nemo_run/code && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"{server_start_cmd} "
    )
    return server_cmd, num_tasks


def get_sandox_command(cluster_config):
    if cluster_config['executor'] == 'none':
        return "python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server"
    return "/entrypoint.sh && /start.sh"


# TODO: Unify the signature of generate.py to use this
def configure_client(
    *,  # Force keyword arguments
    model: str,
    server_type: str,
    server_gpus: int,
    server_nodes: int,
    server_address: str,
    server_port: Optional[int],
    server_args: str,
    extra_arguments: str,
    get_random_port: bool,
):
    """
    Utility function to configure a client for the model inference server.

    Args:
        model: Mounted Path to the model to evaluate.
        server_type: String name of the server type.
        server_address: URL of the server hosting the model.
        server_gpus: Number of GPUs to use for the server.
        server_nodes: Number of nodes to use for the server.
        server_port: Port number for the server.
        server_args: Additional arguments for the server.
        extra_arguments: Extra arguments to pass to the command.
        get_random_port: Whether to get a random port for the server.

    Returns:
        A tuple containing:
            - server_config: Configuration for the server.
            - extra_arguments: Updated extra arguments for the command.
            - server_address: Address of the server.
            - server_port: Port number for the server.
    """
    if server_address is None:  # we need to host the model
        if server_port is None:  # if not specified, we will use a random port or 5000 depending get_random_port
            server_port = get_free_port(strategy="random") if not get_random_port else 5000
        assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
        server_address = f"localhost:{server_port}"

        server_config = {
            "model_path": model,
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "server_args": server_args,
            "server_port": server_port,
        }
        extra_arguments = (
            f"{extra_arguments} ++server.server_type={server_type} "
            f"++server.host=localhost ++server.port={server_port} "
        )
    else:  # model is hosted elsewhere
        server_config = None
        extra_arguments = (
            f"{extra_arguments} ++server.server_type={server_type} "
            f"++server.base_url={server_address} ++server.model={model} "
        )
        server_port = None
    return server_config, extra_arguments, server_address, server_port


@dataclass(kw_only=True)
class CustomJobDetails(SlurmJobDetails):
    # we have 1 srun per sub-task (e.g. server/sandbox/main), but only a single sbatch
    srun_prefix: str = "main"
    sbatch_prefix: str = ""

    @property
    def stdout(self) -> Path:
        return Path(self.folder) / f"{self.sbatch_prefix}%j_sbatch.log"

    @property
    def srun_stdout(self) -> Path:
        return Path(self.folder) / f"{self.srun_prefix}%j_srun.log"

    @property
    def stderr(self) -> Path:
        return Path(self.folder) / f"{self.sbatch_prefix}%j_sbatch.log"

    @property
    def srun_stderr(self) -> Path:
        return Path(self.folder) / f"{self.srun_prefix}%j_srun.log"

    @property
    def ls_term(self) -> str:
        """This term will be used to fetch the logs.

        The command used to list the files is ls -1 {ls_term} 2> /dev/null
        """
        assert self.folder
        return os.path.join(self.folder, "*srun.log")


def read_config(config_file):
    with open(config_file, "rt", encoding="utf-8") as fin:
        cluster_config = yaml.safe_load(fin)

    return cluster_config


def get_cluster_config(cluster=None, config_dir=None):
    """Trying to find an appropriate cluster config.

    Will search in the following order:
    1. config_dir parameter
    2. NEMO_SKILLS_CONFIG_DIR environment variable
    3. Current folder / cluster_configs
    4. This file folder / ../../cluster_configs

    If NEMO_SKILLS_CONFIG is provided and cluster is None,
    it will be used as a full path to the config file
    and NEMO_SKILLS_CONFIG_DIR will be ignored.

    If cluster is a python object (dict-like), then we simply
    return the cluster config, under the assumption that the
    config is prepared by the user.
    """
    # if cluster is provided, we try to find it in one of the folders
    if cluster is not None:
        # check if cluster is a python object instead of a str path, pass through
        if isinstance(cluster, (dict, DictConfig)):
            return cluster

        # either using the provided config_dir or getting from env var
        config_dir = config_dir or os.environ.get("NEMO_SKILLS_CONFIG_DIR")
        if config_dir:
            return read_config(Path(config_dir) / f"{cluster}.yaml")

        # if it's not defined we are trying to find locally
        if (Path.cwd() / 'cluster_configs' / f"{cluster}.yaml").exists():
            return read_config(Path.cwd() / 'cluster_configs' / f"{cluster}.yaml")

        if (Path(__file__).parents[2] / 'cluster_configs' / f"{cluster}.yaml").exists():
            return read_config(Path(__file__).parents[2] / 'cluster_configs' / f"{cluster}.yaml")

        raise ValueError(f"Cluster config {cluster} not found in any of the supported folders.")

    config_file = os.environ.get("NEMO_SKILLS_CONFIG")
    if not config_file:
        LOG.warning(
            "Cluster config is not specified. Running locally without containers. "
            "Only a subset of features is supported and you're responsible "
            "for installing any required dependencies. "
            "It's recommended to run `ns setup` to define appropriate configs!"
        )
        # just returning empty string for any container on access
        cluster_config = {'executor': 'none', 'containers': defaultdict(str)}
        return cluster_config

    if not Path(config_file).exists():
        raise ValueError(f"Cluster config {config_file} not found.")

    cluster_config = read_config(config_file)

    # resolve ssh tunnel config
    if "ssh_tunnel" in cluster_config:
        cluster_config = update_ssh_tunnel_config(cluster_config)

    if cluster_config['executor'] == 'slurm' and "ssh_tunnel" not in cluster_config:
        if "job_dir" not in cluster_config:
            raise ValueError("job_dir must be provided in the cluster config if ssh_tunnel is not provided.")
        set_nemorun_home(cluster_config["job_dir"])

    return cluster_config


def update_ssh_tunnel_config(cluster_config: dict):
    """
    Update the ssh tunnel configuration in the cluster config to resolve job dir and username.
    uses the `user` information to populate `job_dir` if in config.

    Args:
        cluster_config: dict: The cluster configuration dictionary

    Returns:
        dict: The updated cluster configuration dictionary
    """
    if 'ssh_tunnel' not in cluster_config:
        return cluster_config

    resolve_map = [
        dict(key='user', default_env_key='USER'),
        dict(key='job_dir', default_env_key=None),
        dict(key='identity', default_env_key=None),
    ]

    for item in resolve_map:
        key = item['key']
        default_env_key = item['default_env_key']

        if key in cluster_config['ssh_tunnel']:
            # Resolve `user` from env if not provided
            if cluster_config['ssh_tunnel'][key] is None and default_env_key is not None:
                cluster_config['ssh_tunnel'][key] = os.environ[default_env_key]
                LOG.info(f"Resolved `{key}` to `{cluster_config['ssh_tunnel'][key]}`")

            elif isinstance(cluster_config['ssh_tunnel'][key], str) and '$' in cluster_config['ssh_tunnel'][key]:
                cluster_config['ssh_tunnel'][key] = os.path.expandvars(cluster_config['ssh_tunnel'][key])
                LOG.info(f"Resolved `{key}` to `{cluster_config['ssh_tunnel'][key]}`")

    if "$" in cluster_config['ssh_tunnel']['identity']:
        raise ValueError(
            "SSH identity cannot be resolved from environment variables. "
            "Please provide a valid path to the identity file."
        )

    return cluster_config


@lru_cache
def _get_tunnel_cached(
    job_dir: str,
    host: str,
    user: str,
    identity: str | None = None,
    shell: str | None = None,
    pre_command: str | None = None,
):
    return run.SSHTunnel(
        host=host,
        user=user,
        identity=identity,
        shell=shell,
        pre_command=pre_command,
        job_dir=job_dir,
    )


def tunnel_hash(tunnel):
    return f"{tunnel.job_dir}:{tunnel.host}:{tunnel.user}:{tunnel.identity}:{tunnel.shell}:{tunnel.pre_command}"


def get_tunnel(cluster_config):
    if "ssh_tunnel" not in cluster_config:
        if cluster_config["executor"] == "slurm":
            LOG.info("No ssh_tunnel configuration found, assuming we are running from the cluster already.")
        return run.LocalTunnel(job_dir="")
    return _get_tunnel_cached(**cluster_config["ssh_tunnel"])


# Helper class and function to support streaming updates
class OutputWatcher(StreamWatcher):
    """Class for streaming remote tar/compression process."""

    def submit(self, stream):
        print(stream, end='\r')
        sys.stdout.flush()
        return []


def progress_callback(transferred: int, total: int) -> None:
    """Display SFTP transfer progress."""
    percent = (transferred / total) * 100
    bar = '=' * int(percent / 2) + '>'
    sys.stdout.write(
        f'\rFile Transfer Progress: [{bar:<50}] {percent:.1f}% '
        f'({transferred/1024/1024:.1f}MB/{total/1024/1024:.1f}MB)'
    )
    sys.stdout.flush()


def cluster_download(
    tunnel: SSHTunnel, remote_dir: str, local_dir: str, remote_tar_dir: Optional[str] = None, verbose: bool = True
):
    """
    Downloads a directory from a remote cluster by creating a tar archive and transferring it.

    Args:
        tunnel: SSHTunnel connection
        remote_dir: Path to the directory on remote server
        local_dir: Local path to save the downloaded directory
        remote_tar_dir: Optional directory for temporary tar file creation
        verbose: Print download progress
    """

    remote_dir = remote_dir.rstrip('/')
    remote_dir_parent, remote_dir_name = os.path.split(remote_dir)

    # Directory where the remote tarball is written
    remote_tar_dir = remote_tar_dir if remote_tar_dir else remote_dir_parent
    # Path of the remote tar file
    remote_tar_filename = f"{remote_dir_name}.tar.gz"

    # Remote and local tar files
    remote_tar = f"{os.path.join(remote_tar_dir, remote_tar_filename)}"
    local_tar = os.path.join(local_dir, remote_tar_filename)

    # Get the directory size
    result = tunnel.run(f'du -sb {remote_dir} | cut -f1')
    total_size = int(result.stdout.strip())

    # Check if result directory compression is streamable
    streaming_possible = False
    try:
        # Check whether the command pv is present on the remote system or not.
        # Certain systems may not have the `pv` command
        result = tunnel.run('which pv', warn=True)
        streaming_possible = result.exited == 0
    except Exception:
        streaming_possible = False

    if streaming_possible and verbose:
        # We can do streaming compression
        # Command for streaming the compression progress
        command = (
            f'cd {remote_dir_parent} && '
            f'tar --exclude="*.log" -cf - {remote_dir_name} | '
            f'pv -s {total_size} -p -t -e -b -F "Compressing Remote Directory: %b %t %p" | '
            f'gzip > {remote_tar}'
        )
        # Run the remote compression command and stream the progress
        result = tunnel.run(command, watchers=[OutputWatcher()], pty=True, hide=(not verbose))
    else:
        command = f'cd {remote_dir_parent} && tar -czf {remote_tar} {remote_dir_name}'
        result = tunnel.run(command, hide=(not verbose))

    # Get SFTP client from tunnel's session's underlying client
    sftp = tunnel.session.client.open_sftp()

    # Use SFTP's get with callback
    sftp.get(remote_tar, local_tar, callback=progress_callback if verbose else None)
    print(f"\nTransfer complete: {local_tar}")

    # Extract the tarball locally
    os.makedirs(local_dir, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=local_dir)

    # Clean up the tarball from the remote server
    tunnel.run(f'rm {remote_tar}', hide=True)

    # Clean up the local tarball
    os.remove(local_tar)


def cluster_upload(tunnel: SSHTunnel, local_file: str, remote_dir: str, verbose: bool = True):
    """
    Uploads a file to cluster.
    TODO: extend to a folder.

    Args:
        tunnel: SSHTunnel connection
        local_file: Path to the local file to upload
        remote_dir: Cluster path where to save the file
        verbose: Print upload progress
    """
    sftp = tunnel.session.client.open_sftp()
    sftp.put(str(local_file), str(remote_dir), callback=progress_callback if verbose else None)
    print(f"\nTransfer complete")


def get_git_repo_path(path: str | Path = None):
    """Check if the path is a git repo.

    Args:
        path: Path to the directory to check. If None, will check the current directory.

    Returns:
        Path to the repo if it is a git repo, otherwise None.
    """
    original_path = os.getcwd()
    try:
        if path:
            os.chdir(path)

        repo_path = (
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        return Path(repo_path)

    except subprocess.CalledProcessError:
        return None

    finally:
        os.chdir(original_path)


def get_packager(extra_package_dirs: tuple[str] | None = None):
    """Will check if we are running from a git repo and use git packager or default packager otherwise."""
    nemo_skills_dir = get_registered_external_repo('nemo_skills').path

    if extra_package_dirs:
        include_patterns = [str(Path(d) / '*') for d in extra_package_dirs]
        include_pattern_relative_paths = [str(Path(d).parent) for d in extra_package_dirs]
    else:
        include_patterns = []
        include_pattern_relative_paths = []

    check_uncommited_changes = not bool(os.getenv('NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK', 0))

    # are we in a git repo? If yes, we are uploading the current code
    repo_path = get_git_repo_path(path=None)  # check if we are in a git repo in pwd

    if repo_path:
        # Do we have nemo_skills package in this repo? If no, we need to pick it up from installed location
        if not (Path(repo_path) / 'nemo_skills').is_dir():
            logging.info(
                "Not running from NeMo-Skills repo, trying to upload installed package. "
                "Make sure there are no extra files in %s",
                str(nemo_skills_dir / '*'),
            )
            include_patterns.append(str(nemo_skills_dir / '*'))
        else:
            # picking up local dataset files if we are in the right repo
            include_patterns.append(str(nemo_skills_dir / "dataset/**/*.jsonl"))
        include_pattern_relative_paths.append(str(nemo_skills_dir.parent))

        root_package = run.GitArchivePackager(
            include_pattern=include_patterns,
            include_pattern_relative_path=include_pattern_relative_paths,
            check_uncommitted_changes=check_uncommited_changes,
        )
    else:
        logging.info(
            "Not running from a git repo, trying to upload installed package. Make sure there are no extra files in %s",
            str(nemo_skills_dir / '*'),
        )
        include_patterns.append(str(nemo_skills_dir / '*'))
        include_pattern_relative_paths.append(str(nemo_skills_dir.parent))

        root_package = run.PatternPackager(
            include_pattern=include_patterns,
            relative_path=include_pattern_relative_paths,
        )

    extra_repos = {}
    if len(EXTERNAL_REPOS) > 1:
        # Insert root package as the first package
        extra_repos['nemo_run'] = root_package

        for repo_name, repo_meta in EXTERNAL_REPOS.items():
            if repo_name == 'nemo_skills':
                continue

            repo_path = repo_meta.path
            if get_git_repo_path(repo_path):
                # Extra repos is a git repos, so we need to package only committed files
                extra_repos[repo_name] = run.GitArchivePackager(
                    basepath=str(repo_path), check_uncommitted_changes=check_uncommited_changes
                )
            else:
                # Extra repos is not a git repo, so we need to package all files in the directory
                repo_include_pattern = [str(Path(repo_path) / '*')]
                repo_include_pattern_relative_path = [str(Path(repo_path).parent)]
                extra_repos[repo_name] = run.PatternPackager(
                    include_pattern=repo_include_pattern,
                    relative_path=repo_include_pattern_relative_path,
                )

        # Return hybrid packager
        return run.HybridPackager(sub_packagers=extra_repos, extract_at_root=True)

    return root_package


def get_env_variables(cluster_config):
    """
    Will get the environment variables from the cluster config and the user environment.

    The following items in the cluster config are supported:
    - `required_env_vars` - list of required environment variables
    - `env_vars` - list of optional environment variables

    WANDB_API_KEY, NVIDIA_API_KEY, OPENAI_API_KEY, and HF_TOKEN are always added if they exist.

    Args:
        cluster_config: cluster config dictionary

    Returns:
        dict: dictionary of environment
    """
    env_vars = {}
    # Check for user requested env variables
    required_env_vars = cluster_config.get("required_env_vars", [])
    for env_var in required_env_vars:
        if "=" in env_var:
            if env_var.count("=") == 1:
                env_var, value = env_var.split("=")
            else:
                raise ValueError(f"Invalid required environment variable format: {env_var}")
            env_vars[env_var.strip()] = value.strip()
            logging.info(f"Adding required environment variable {env_var}")
        elif env_var in os.environ:
            logging.info(f"Adding required environment variable {env_var} from environment")
            env_vars[env_var] = os.environ[env_var]
        else:
            raise ValueError(f"Required environment variable {env_var} not found.")

    # It is fine to have these as always optional even if they are required for some configs
    # Assume it is required, then this will override the value set above with the same
    # value, assuming it has not been updated externally between these two calls
    always_optional_env_vars = ["WANDB_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"]
    default_factories = {
        "HF_TOKEN": lambda: str(get_token()),
    }
    # Add optional env variables
    optional_env_vars = cluster_config.get("env_vars", [])
    for env_var in optional_env_vars + always_optional_env_vars:
        if "=" in env_var:
            if env_var.count("=") == 1:
                env_var, value = env_var.split("=")
            else:
                raise ValueError(f"Invalid optional environment variable format: {env_var}")
            env_vars[env_var.strip()] = value.strip()
            logging.info(f"Adding optional environment variable {env_var}")
        elif env_var in os.environ:
            logging.info(f"Adding optional environment variable {env_var} from environment")
            env_vars[env_var] = os.environ[env_var]
        elif env_var in default_factories:
            env_vars[env_var] = default_factories[env_var]()
            logging.info(f"Adding optional environment variable {env_var} from environment")
        else:
            logging.info(f"Optional environment variable {env_var} not found in user environment; skipping.")

    return env_vars


def wrap_cmd(cmd, preprocess_cmd, postprocess_cmd, random_seed=None, wandb_parameters=None):
    if preprocess_cmd:
        if random_seed is not None:
            preprocess_cmd = preprocess_cmd.format(random_seed=random_seed)
        cmd = f" {preprocess_cmd} && {cmd} "
    if postprocess_cmd:
        if random_seed is not None:
            postprocess_cmd = postprocess_cmd.format(random_seed=random_seed)
        cmd = f" {cmd} && {postprocess_cmd} "
    if wandb_parameters:
        log_wandb_cmd = (
            f"python -m nemo_skills.inference.log_samples_wandb "
            f"    {wandb_parameters['samples_file']} "
            f"    --name={wandb_parameters['name']} "
            f"    --project={wandb_parameters['project']} "
        )
        if wandb_parameters['group'] is not None:
            log_wandb_cmd += f" --group={wandb_parameters['group']} "
        cmd = f"{cmd} && {log_wandb_cmd} "
    return cmd


def get_mounts_from_config(cluster_config: dict):
    """
    Determines if there are mount paths that are being passed via environment variables.
    Selects the key in the cluster config called `mounts` which is a list of strings.
    Each string is in the format of `<str | {env_var}>:<str | {env_var}>` where `env_var`
    is the name of the environment variable.

    Args:
        cluster_config (dict): cluster config dictionary

    Returns:
        list: updated list of mounts
    """
    mounts = cluster_config.get('mounts', [])

    # if there are env_mounts, we will add the mounts from the env_mounts
    for mount_id in range(len(mounts)):
        mount = mounts[mount_id]

        if ":" not in mount:
            raise ValueError(f"Invalid mount format: {mount}. The mount path must be separated by a colon.")

        mount_source, mount_target = mount.split(":")

        if mount_source[0] == "{" and mount_source[-1] == "}":
            # Resolve the environment variable for the mount source
            mount_source = mount_source[1:-1]

            if mount_source not in os.environ:
                raise ValueError(
                    f"Required environment variable {mount_source} not found in env variables passed in cluster configs."
                )

            mount_source = os.environ[mount_source]

        if mount_target[0] == "{" and mount_target[-1] == "}":
            # Resolve the environment variable for the mount target
            mount_target = mount_target[1:-1]

            if mount_target not in os.environ:
                raise ValueError(
                    f"Required environment variable {mount_target} not found in env variables passed in cluster configs."
                )

            mount_target = os.environ[mount_target]

        # add the mount to the list of mounts
        resolved_mount = f"{mount_source}:{mount_target}"
        mounts[mount_id] = resolved_mount

    return mounts


def get_executor(
    cluster_config,
    container,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    job_name,
    log_dir,
    log_prefix: str = "main",
    mounts=None,
    partition=None,
    time_min=None,
    dependencies=None,
    extra_package_dirs: tuple[str] | None = None,
    heterogeneous=False,
    het_group=None,
    total_het_groups=None,
    slurm_kwargs: dict | None = None,
):
    env_vars = get_env_variables(cluster_config)
    config_mounts = get_mounts_from_config(cluster_config)

    mounts = mounts or config_mounts
    if extra_package_dirs is not None:
        extra_package_dirs = tuple(extra_package_dirs)
    packager = get_packager(extra_package_dirs=extra_package_dirs)

    if cluster_config["executor"] != "slurm":
        if num_nodes > 1:
            raise ValueError("Local executor does not support multi-node execution")

    if cluster_config["executor"] == "none":
        return LocalExecutor()

    if cluster_config["executor"] == "local":
        env_vars["PYTHONUNBUFFERED"] = "1"  # this makes sure logs are streamed right away
        return DockerExecutor(
            container_image=container,
            packager=packager,
            ipc_mode="host",
            volumes=mounts,
            ntasks_per_node=1,
            # locally we are always asking for all GPUs to be able to select a subset with CUDA_VISIBLE_DEVICES
            num_gpus=-1 if gpus_per_node is not None else None,
            network="host",
            env_vars=env_vars,
            additional_kwargs={"entrypoint": ""},
        )

    if not heterogeneous:
        env_vars["SLURM_MASTER_NODE"] = "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
    else:
        # master node will be within the same group
        env_vars["SLURM_MASTER_NODE"] = (
            f"$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_{het_group} | head -n1)"
        )
        # in addition defining master nodes for all groups to allow communication
        for group in range(total_het_groups):
            env_vars[f"SLURM_MASTER_NODE_HET_GROUP_{group}"] = (
                f"$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_{group} | head -n1)"
            )

    partition = partition or cluster_config.get("partition")
    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition]

    additional_parameters = {'time_min': time_min} if time_min is not None else {}
    if cluster_config.get('mail_type') is not None:
        additional_parameters['mail_type'] = cluster_config['mail_type']
    if cluster_config.get('mail_user') is not None:
        additional_parameters['mail_user'] = cluster_config['mail_user']
    srun_args = [
        "--no-container-mount-home",
        "--overlap",
        "--mpi=pmix",
        '--wait=10',
        # we need to be explicit about this in srun as commands might need to run in parallel
        f"--ntasks-per-node={tasks_per_node}",
        f"--nodes={num_nodes}",
        # NeMo-run should take care of this, but we'll put it here temporarily
        f"--container-env={','.join([k.strip() for k in env_vars.keys()])}",
    ]
    if not cluster_config.get("disable_gpus_per_node", False) and gpus_per_node is not None:
        srun_args.append(f"--gpus-per-node={gpus_per_node}")

    dependency_type = cluster_config.get("dependency_type", "afterany")

    return run.SlurmExecutor(
        account=cluster_config["account"],
        partition=partition,
        nodes=num_nodes,
        ntasks_per_node=tasks_per_node,
        tunnel=get_tunnel(cluster_config),
        container_image=container,
        container_mounts=mounts,
        time=timeout,
        additional_parameters=additional_parameters,
        packager=packager,
        gpus_per_node=gpus_per_node if not cluster_config.get("disable_gpus_per_node", False) else None,
        srun_args=srun_args,
        job_details=CustomJobDetails(
            job_name=cluster_config.get("job_name_prefix", "") + job_name,
            folder=get_unmounted_path(cluster_config, log_dir),
            srun_prefix=log_prefix + '_' + job_name + '_',
            sbatch_prefix=job_name + '_',
        ),
        wait_time_for_group_job=0.01,
        monitor_group_job_wait_time=20,
        dependencies=dependencies,
        dependency_type=dependency_type,
        heterogeneous=heterogeneous,
        env_vars=env_vars,
        **(slurm_kwargs or {}),
    )


@contextmanager
def temporary_env_update(cluster_config, updates):
    original_env_vars = cluster_config.get("env_vars", []).copy()
    updated_env_vars = original_env_vars.copy()
    for key, value in updates.items():
        updated_env_vars.append(f"{key}={value}")
        cluster_config["env_vars"] = updated_env_vars
    try:
        yield
    finally:
        cluster_config["env_vars"] = original_env_vars


# TODO: this function has become too cumbersome to use with all recently added support
#       we should make it simpler by perhaps removing separate logic for server/sandbox
#       and supporting them through a list of cmds directly
#       should also make heterogenous logic very clear and more robust
#       and all parameters that can be list should be list for consistency
def add_task(
    exp,
    cmd: str | list[str],
    task_name,
    cluster_config,
    container: str | list[str],
    num_tasks: int | list[int] = 1,
    num_gpus=None,
    num_nodes=1,
    log_dir=None,
    partition=None,
    time_min=None,
    with_sandbox=False,
    sandbox_port: int | None = None,
    server_config=None,
    reuse_code_exp: str | run.Experiment | None = None,
    reuse_code: bool = True,
    task_dependencies: list[str] = None,
    run_after: str | list[str] | None = None,
    get_server_command=get_server_command,
    extra_package_dirs: list[str] | None = None,
    slurm_kwargs: dict | None = None,
    heterogeneous: bool = False,
):
    """Wrapper for nemo-run exp.add to help setting up executors and dependencies.

    Note that there are two parameters that control dependencies.
        - task_dependencies: list of tasks that this task depends on **within the same experiment**
        - run_after: a string with experiment name or a list of experiment names that this task
          should run after. Will schedule dependencies on all tasks inside `run_after` experiments.
          It needs to already be launched and running.

    Example of how to set task_dependencies:

    with get_exp(expname, cluster_config) as exp:
        task1 = add_task(exp, ...)
        task2 = add_task(exp, ..., task_dependencies=[task1])

    You can use `reuse_code_exp` to reuse the code from another experiment
    (and thus avoid costly packaging/ssh uploading). You can provide either experiment
    name or the experiment object itself.

    By default we will reuse the code of the first submitted experiment.
    If you want to avoid this, set `reuse_code=False`.
    """
    if run_after is not None and cluster_config["executor"] == "slurm":
        if isinstance(run_after, (str, run.Experiment)):
            run_after = [run_after]
        dependencies = []
        for dep_expname in run_after:
            exp_handles = get_exp_handles(dep_expname)
            if len(exp_handles) == 0:
                LOG.warning(
                    "No pending or running tasks found for experiment %s, cannot set dependencies.", dep_expname
                )
            dependencies.extend(exp_handles)
        if len(dependencies) == 0:
            dependencies = None
    else:
        dependencies = None

    if num_gpus is None and cluster_config['executor'] == "slurm":
        if not 'cpu' in (partition or cluster_config.get("partition", "")):
            num_gpus = 1

    if sandbox_port is None:
        sandbox_port = get_free_port(strategy="random")

    het_group = 0
    het_group_indices = []
    total_het_groups = (server_config is not None) + bool(cmd) + with_sandbox

    commands = []
    executors = []
    # assuming server always has the largest resources request, so it needs to go first
    if server_config is not None:
        server_cmd, num_server_tasks = get_server_command(**server_config, cluster_config=cluster_config)
        if 'container' not in server_config:
            server_container = cluster_config["containers"][server_config['server_type']]
        server_executor = get_executor(
            cluster_config=cluster_config,
            container=server_container,
            num_nodes=server_config['num_nodes'],
            tasks_per_node=num_server_tasks,
            gpus_per_node=server_config['num_gpus'],
            partition=partition,
            time_min=time_min,
            dependencies=dependencies,
            job_name=task_name,
            log_dir=log_dir,
            log_prefix="server",
            extra_package_dirs=extra_package_dirs,
            slurm_kwargs=slurm_kwargs,
            heterogeneous=heterogeneous,
            het_group=het_group,
            total_het_groups=total_het_groups,
        )
        if cluster_config["executor"] != "slurm" and num_server_tasks > 1:
            server_cmd = f"mpirun --allow-run-as-root -np {num_server_tasks} bash -c {shlex.quote(server_cmd)}"
        commands.append(server_cmd)
        executors.append(server_executor)
        het_group_indices.append(het_group)
        het_group += 1

    # then goes the main task(s) unless it's empty
    if cmd:
        if isinstance(cmd, str):
            cmd = [cmd]
        if isinstance(container, str):
            container = [container]
        if isinstance(num_tasks, int):
            num_tasks = [num_tasks]
        if len(cmd) != len(container) or len(cmd) != len(num_tasks):
            raise ValueError("Number of commands, containers and num_tasks must match.")
        for cur_idx, (cur_cmd, cur_container, cur_tasks) in enumerate(zip(cmd, container, num_tasks)):
            if cluster_config["executor"] != "slurm" and cur_tasks > 1:
                cur_cmd = f"mpirun --allow-run-as-root -np {cur_tasks} bash -c {shlex.quote(cur_cmd)}"
            with temporary_env_update(cluster_config, {"NEMO_SKILLS_SANDBOX_PORT": sandbox_port}):
                commands.append(cur_cmd)
                executors.append(
                    get_executor(
                        cluster_config=cluster_config,
                        container=cur_container,
                        num_nodes=num_nodes,
                        tasks_per_node=cur_tasks,
                        gpus_per_node=num_gpus,
                        partition=partition,
                        time_min=time_min,
                        dependencies=dependencies,
                        job_name=task_name,
                        log_dir=log_dir,
                        log_prefix="main" if len(cmd) == 1 else f"main_{cur_idx}",
                        extra_package_dirs=extra_package_dirs,
                        slurm_kwargs=slurm_kwargs,
                        heterogeneous=heterogeneous,
                        het_group=het_group,
                        total_het_groups=total_het_groups,
                    )
                )
                het_group_indices.append(het_group)
        het_group += 1

    # finally a sandbox if needed
    if with_sandbox:
        sandbox_env_updates = {"LISTEN_PORT": sandbox_port}
        current_env_vars = cluster_config.get("env_vars", []).copy()
        for override in current_env_vars:
            if "PYTHONPATH" in override:
                if override.startswith("PYTHONPATH="):
                    override = override[11:]
                sandbox_env_updates["PYTHONPATH"] = override + ":/app"

        with temporary_env_update(cluster_config, sandbox_env_updates):
            commands.append(get_sandox_command(cluster_config))
            sandbox_executor = get_executor(
                cluster_config=cluster_config,
                container=cluster_config["containers"]["sandbox"],
                num_nodes=executors[0].nodes if cluster_config["executor"] == "slurm" else 1,
                tasks_per_node=1,
                gpus_per_node=num_gpus,
                partition=partition,
                time_min=time_min,
                mounts=tuple(),  # we don't want to mount anything
                dependencies=dependencies,
                job_name=task_name,
                log_dir=log_dir,
                log_prefix="sandbox",
                extra_package_dirs=extra_package_dirs,
                slurm_kwargs=slurm_kwargs,
                heterogeneous=heterogeneous,
                het_group=het_group,
                total_het_groups=total_het_groups,
            )
            executors.append(sandbox_executor)
            het_group_indices.append(het_group)
        het_group += 1

    if cluster_config["executor"] != "local":
        tunnel = get_tunnel(cluster_config)
        if isinstance(tunnel, run.SSHTunnel) and reuse_code:
            reuse_code_exp = reuse_code_exp or REUSE_CODE_EXP.get(tunnel_hash(tunnel))
            if reuse_code_exp is not None:
                if isinstance(reuse_code_exp, str):
                    try:
                        reuse_code_exp = run.Experiment.from_id(reuse_code_exp)
                    except Exception:
                        LOG.debug(f"Failed to create experiment from id {reuse_code_exp}, trying to find it by title")
                        reuse_code_exp = run.Experiment.from_title(reuse_code_exp)

                LOG.info("Trying to reuse code from experiment %s", reuse_code_exp._title)
                reuse_key = get_packaging_job_key(reuse_code_exp._id, "nemo-run")
                if reuse_key in reuse_code_exp.tunnels[tunnel.key].packaging_jobs:
                    reuse_dir = reuse_code_exp.tunnels[tunnel.key].packaging_jobs[reuse_key].dst_path

                    for executor in executors:
                        executor.packager.symlink_from_remote_dir = reuse_dir
                    LOG.info(f"Successfully reused code from {reuse_key}")
                else:
                    LOG.warning("Relevant packaging job not found for experiment %s", reuse_code_exp._title)
        # if current is not reused, we are refreshing the cache as there is a reason to believe it's outdated
        elif isinstance(tunnel, run.SSHTunnel):
            REUSE_CODE_EXP.pop(tunnel_hash(tunnel), None)

    # no mounting here, so assuming /nemo_run/code can be replaced with the current dir
    if cluster_config["executor"] == "none":
        for idx in range(len(commands)):
            commands[idx] = commands[idx].replace('/nemo_run/code', './')

    if len(commands) == 1:
        # to keep sbatch script simpler, we don't wrap in a list in this case
        return exp.add(
            run.Script(inline=commands[0]),
            executor=executors[0],
            name="nemo-run",
            dependencies=task_dependencies,
        )
    else:
        if heterogeneous:
            executors[0].het_group_indices = het_group_indices
        return exp.add(
            [run.Script(inline=command) for command in commands],
            executor=executors,
            name="nemo-run",
            dependencies=task_dependencies,
        )


def run_exp(exp, cluster_config, sequential=None):
    """If sequential is not specified, using True locally and False otherwise.

    If it is specified, it will be used as is.
    """
    if cluster_config['executor'] != 'slurm':
        exp.run(detach=False, tail_logs=True, sequential=True if sequential is None else sequential)
    else:
        exp.run(detach=True, sequential=False if sequential is None else sequential)

        # caching the experiment code for reuse
        tunnel = get_tunnel(cluster_config)
        if isinstance(tunnel, run.SSHTunnel):
            ssh_hash = tunnel_hash(tunnel)
            if ssh_hash not in REUSE_CODE_EXP:
                REUSE_CODE_EXP[ssh_hash] = exp


def get_exp(expname, cluster_config):
    if cluster_config['executor'] == 'slurm':
        return run.Experiment(expname)
    # hiding all nemo-run logs otherwise as they are not useful locally
    if cluster_config['executor'] == 'local':
        return run.Experiment(expname, clean_mode=True)
    return run.Experiment(expname, clean_mode=True, log_level="WARN")

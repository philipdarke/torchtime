import os
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch

from torchtime.constants import EPS


def sample_indices(length, proportion, generator=None, seed=None):
    """Vector indices to select ``proportion`` from a batch of size ``length``."""
    if generator is None:
        generator = torch.Generator()
        if seed is not None:
            generator = generator.manual_seed(seed)
    subset_size = int(length * proportion)
    return torch.randperm(length, generator=generator)[:subset_size]


def _download_archive(url, path):
    """Download and extract a .zip/.tar.gz file from ``url`` to ``path``."""
    print_message("Downloading " + url + "...")
    with tempfile.NamedTemporaryFile() as temp_file:
        download = requests.get(url)
        temp_file.write(download.content)
        url_path = urlparse(url).path
        if url_path[-3:] == "zip":
            zipfile.ZipFile(temp_file, "r").extractall(path)
        elif url_path[-6:] == "tar.gz" or url_path[-3:] == "tgz":
            tarfile.open(temp_file.name).extractall(path)
        else:
            assert "file type not supported"


def download_file(url, path):
    "Download a file from ``url`` and save to ``path``."
    if not path.is_dir():
        os.makedirs(path)
    url_path = urlparse(url).path
    file_name = Path(url_path).name
    file_path = path / file_name
    if not file_path.is_file():
        with open(file_path, "wb") as save_file:
            print_message("Downloading " + url + "...")
            download = requests.get(url)
            save_file.write(download.content)


def physionet_download(urls, path):
    """Download and extract .zip/.tar.gz files if not already downloaded. ``urls`` must
    be a dictionary in format ``{[folder]: url}`` where ``folder`` is the name of the
    extracted folder."""
    for dataset in urls:
        if not (path / dataset).is_dir():
            _download_archive(urls[dataset], path)


def get_file_list(data_directories, proportion=1, seed=None):
    """Get list of files in a directory. ``data_directories`` is a list of the
    directories to scan. The ``proportion`` argument returns a random subset of the
    files in each directory."""
    all_files = [None for _ in data_directories]
    for i, directory in enumerate(data_directories):
        data_files = np.sort(np.array(os.listdir(directory)))
        if proportion < (1.0 - EPS):
            idx = sample_indices(len(data_files), proportion, seed=seed).numpy()
            data_files = data_files[idx]
        all_files[i] = list(data_files)
    return all_files


def print_message(message, type="message"):
    if type == "message":
        colour = "\033[33m"
    elif type == "error":
        colour = "\033[31m"
    elif type == "info":
        colour = "\033[32m"
    else:
        assert "argument 'type' must be 'message', 'error' or 'info'"
    print(colour, message, "\033[m", sep="")


def nan_mode(input):
    """Mode value for tensor (ignoring NaNs)."""
    return torch.mode(input[~torch.isnan(input)])[0]

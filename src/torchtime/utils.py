import hashlib
import os
import re
import tarfile
import tempfile
import zipfile
from pathlib import Path, PosixPath
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from tqdm import tqdm

from torchtime.constants import CHECKSUM_EXT, DATASET_OBJS, OBJ_EXT, TQDM_FORMAT

# Utilities ----------------------------------------------------------------------------


def _nanmode(input):
    """Mode value for tensor ignoring ``NaN``s."""
    return torch.mode(input[~torch.isnan(input)])[0]


# Sampling -----------------------------------------------------------------------------


def _generator(seed=None):
    """Returns a PyTorch generator object seeded with optional seed."""
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator = generator.manual_seed(seed)
    return generator


def _sample_indices(length, proportion, generator=None, seed=None):
    """Vector indices to select ``proportion`` from a batch of size ``length``."""
    if generator is None:
        generator = _generator(seed)
    else:
        if seed is not None:
            generator = generator.manual_seed(seed)
    subset_size = int(length * proportion)
    return torch.randperm(length, generator=generator)[:subset_size]


def _get_file_list(directories):
    """Returns a list of files in ``directories``."""
    if type(directories) is str or type(directories) is PosixPath:
        directories = [directories]
    all_files = [None for _ in directories]
    for i, directory in enumerate(directories):
        data_files = [obj for obj in Path(directory).iterdir() if obj.is_file()]
        data_files = np.sort(np.array(data_files))
        all_files[i] = list(data_files)
    if len(all_files) == 1:
        all_files = all_files[0]
    return all_files


def _simulate_missing(X, missing, generator=None, seed=None):
    """Simulate missing data by modifying ``X`` in place."""
    length = X.size(1)
    if generator is None:
        generator = _generator(seed)
    else:
        if seed is not None:
            generator = generator.manual_seed(seed)
    for Xi in X:
        if type(missing) in [int, float]:
            idx = _sample_indices(length, missing, generator)
            Xi[idx] = float("nan")
        else:
            assert Xi.size(-1) == len(
                missing
            ), "argument 'missing' must be same length as number of channels \
                ({})".format(
                Xi.size(-1)
            )
            for channel, rate in enumerate(missing):
                idx = _sample_indices(length, rate, generator)
                Xi[idx, channel] = float("nan")


# Download data ------------------------------------------------------------------------


def _validate_url(url):
    """Regex from
    https://github.com/django/django/blob/stable/1.3.x/django/core/validators.py#L45."""
    regex = re.compile(
        r"^(?:http|ftp)s?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|"
        r"[A-Z0-9-]{2,}\.?)|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)",
        re.IGNORECASE,
    )
    assert re.match(regex, url) is not None, "{} is not a valid URL".format(url)
    return url


def _get_url_filename(url):
    """Extract file name from URL."""
    url = _validate_url(url)
    url_path = urlparse(url).path
    assertion_error = "{} does not point to a file".format(url)
    assert url_path != "", assertion_error
    assert url_path[-1] != "/", assertion_error
    file_name = Path(url_path).name
    assert file_name != "", assertion_error
    return file_name


def _download_object(url, save_file):
    """Download a file to ``save_file``."""
    try:
        print("Downloading " + url + "...")
        download = requests.get(url, stream=True)
        download_size = int(download.headers.get("content-length", 0))
        with tqdm(
            total=download_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            delay=1,
            bar_format=TQDM_FORMAT,
        ) as bar:
            for chunk in download.iter_content(chunk_size=1024):
                size = save_file.write(chunk)
                save_file.flush()
                bar.update(size)
            download.close()
    except Exception:
        raise Exception(
            "could not download, check URL ({}) and internet access".format(url)
        )


def _download_to_file(url, file_path, overwrite=False):
    """Download a file and save as ``file_path`` if not already downloaded. Use
    ``overwrite`` to force a new download."""
    if type(file_path) is tempfile._TemporaryFileWrapper:
        _download_object(url, file_path)
    else:
        file_path = Path(file_path)
        if overwrite or not file_path.is_file():
            if not file_path.parent.is_dir():
                os.mkdir(file_path.parent)
            with open(file_path, "wb") as save_file:
                _download_object(url, save_file)


def _download_to_directory(url, path, overwrite=False):
    """Download a file to the ``path`` directory if not already downloaded. Use
    ``override`` to force a new download."""
    file_name = _get_url_filename(url)
    path = Path(path)
    if not path.is_dir():
        os.mkdir(path)
    _download_to_file(url, path / file_name, overwrite)


def _download_archive(url, path):
    """Download and extract a ``.zip``/``.tar.gz`` file from ``url`` to the ``path``
    directory."""
    file_name = _get_url_filename(url)
    file_ext = Path(file_name).suffix
    if file_ext in [".zip", ".gz", ".tgz"]:
        try:
            with tempfile.NamedTemporaryFile() as temp_file:
                _download_to_file(url, temp_file)
                if file_ext == ".zip":
                    zipfile.ZipFile(temp_file, "r").extractall(path)
                else:
                    tarfile.open(temp_file.name).extractall(path)
        except Exception:
            raise Exception(
                "could not extract archive, check URL ({}) and internet access".format(
                    url
                )
            )
    else:
        raise Exception("file type ({}) is unsupported".format(file_ext))


def _physionet_download(urls, path, overwrite=False):
    """Download and extract ``.zip``/``.tar.gz`` files if download folder is not
    present. ``urls`` must be a dictionary in format ``{directory: url}`` where
    ``directory`` is the name of the extracted directory. Use ``override`` to force a
    new download."""
    for dataset in urls:
        if not (path / dataset).is_dir() or overwrite:
            _download_archive(urls[dataset], path)


# Checksums ----------------------------------------------------------------------------


def _get_SHA256(file):
    """Get SHA256 for file."""
    with open(file, "rb") as check_file:
        checksum = hashlib.sha256()
        chunk = check_file.read(8192)
        while chunk:
            checksum.update(chunk)
            chunk = check_file.read(8192)
    return checksum.hexdigest()


def _check_SHA256(file, check_file):
    """Check file against ``.sha256`` file."""
    sha = _get_SHA256(file)
    with open(check_file, "rb") as f:
        file_sha = f.read().decode()
    return sha == file_sha


# Cache data ---------------------------------------------------------------------------


def _cache_exists(path):
    """Check cached data exists."""
    return all(
        [
            (path / (obj + OBJ_EXT)).is_file()
            and (path / (obj + CHECKSUM_EXT)).is_file()
            for obj in DATASET_OBJS
        ]
    )


def _validate_cache(path):
    """Validate checksums for cache."""
    print("Validating cache...")
    valid_cache = True
    for obj in DATASET_OBJS:
        file_path = path / (obj + OBJ_EXT)
        sha_path = path / (obj + CHECKSUM_EXT)
        if not _check_SHA256(file_path, sha_path):
            print("SHA256 check failed for {}!".format(file_path))
            valid_cache = False
            break
    return valid_cache


def _cache_data(path, X, y, length):
    """Cache tensors and checksums."""
    # Make cache directory
    if not path.is_dir():
        os.makedirs(path)
    # Save objects
    for i, obj in enumerate([X, y, length]):
        obj_path = path / (DATASET_OBJS[i] + OBJ_EXT)
        sha_path = path / (DATASET_OBJS[i] + CHECKSUM_EXT)
        torch.save(obj, obj_path)
        with open(sha_path, "w") as f:
            f.write(_get_SHA256(obj_path))

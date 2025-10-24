import h5py
import numpy as np
from h5py import File


def load_packed_datasets(
    idx: int,
    chunk_size: int,
    h5_file: File,
    target_dataset_name: str,
    cumulative_lengths: np.ndarray,
) -> np.ndarray:
    """Load a packed dataset from an HDF5 file given an index and chunk size.

    Args:
        idx: The index of the chunk to load.
        chunk_size: The size of each chunk.
        h5_file: The HDF5 file object.
        target_dataset_name: The name of the target dataset in the HDF5 file.
        cumulative_lengths: Cumulative lengths of sequences.

    Returns:
        array: The loaded data as a numpy array, in a packed format, consisting of
            chunk_size sequences.
    """
    start_idx = cumulative_lengths[idx]
    end_idx = cumulative_lengths[idx + chunk_size]
    dataset = h5_file[target_dataset_name]
    assert isinstance(dataset, h5py.Dataset)
    return dataset[start_idx:end_idx]


def load_fixed_datasets(
    idx: int,
    chunk_size: int,
    h5_file: File,
    target_dataset_name: str,
) -> np.ndarray:
    """Load a fixed-size dataset from an HDF5 file given an index and chunk size.

    Args:
        idx: The index of the chunk to load.
        chunk_size: The size of each chunk.
        h5_file: The HDF5 file object.
        target_dataset_name: The name of the target dataset in the HDF5 file.

    Returns:
        array: The loaded data as a numpy array, of shape (chunk_size, ...).
    """
    start_idx = idx
    end_idx = idx + chunk_size
    dataset = h5_file[target_dataset_name]
    assert isinstance(dataset, h5py.Dataset)
    return dataset[start_idx:end_idx]

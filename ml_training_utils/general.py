from typing import Tuple, List, Optional, Union, Dict
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import shutil

from pathlib import Path
import shutil
import json
from typing import Set, List, Optional


def remove_non_complete_runs(base_dir: str, required_files: Optional[Set[str]] = None) -> List[str]:
    """
    Removes experiment runs that don't have all required files/folders.

    Args:
        base_dir: Base directory containing experiment runs
        required_files: Set of required files/folders. Defaults to {'model.pt', 'metadata.json'}

    Returns:
        List of removed run IDs

    Raises:
        FileNotFoundError: If base_dir doesn't exist
        ValueError: If base_dir doesn't contain runs_index.json
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    index_path = base_path / "runs_index.json"
    if not index_path.exists():
        raise ValueError(f"No runs_index.json found in {base_dir}")

    required_files = required_files or {'model.pt', 'metadata.json'}
    removed_runs = []

    # Load runs index
    with open(index_path) as f:
        runs_index = json.load(f)

    # Check each run directory
    for run_id, metadata in list(runs_index.items()):
        run_path = Path(metadata['run_folder'])

        # Remove if directory doesn't exist
        if not run_path.exists():
            del runs_index[run_id]
            removed_runs.append(run_id)
            continue

        # Check for required files
        missing_files = [f for f in required_files
                         if not (run_path / f).exists()]

        if missing_files:
            try:
                shutil.rmtree(run_path)
                del runs_index[run_id]
                removed_runs.append(run_id)
            except OSError as e:
                print(f"Error removing {run_path}: {e}")

    # Update runs index
    with open(index_path, 'w') as f:
        json.dump(runs_index, f, indent=4)

    return removed_runs


class TimeWindowSplitter:
    def __init__(self, data: torch.Tensor, index_to_info: Dict, window_size: int):
        if data.ndim < 2:
            raise ValueError("Data must have shape [samples, ..., time]")
        if window_size <= 0:
            raise ValueError("Window size must be positive")

        self.data = data
        self.index_to_info = index_to_info
        self.window_size = window_size
        self.original_shape = data.shape

    def split(self) -> Tuple[torch.Tensor, Dict]:
        samples = self.data.shape[0]
        *spatial_dims, time = self.data.shape[1:]

        if time % self.window_size != 0:
            raise ValueError(f"Time dimension ({time}) must be divisible by window size ({self.window_size})")

        num_windows = time // self.window_size
        new_shape = (samples * num_windows, *spatial_dims, self.window_size)
        new_data = torch.zeros(new_shape, dtype=self.data.dtype, device=self.data.device)

        for i in range(samples):
            for j in range(num_windows):
                new_idx = i * num_windows + j
                start_idx = j * self.window_size
                end_idx = start_idx + self.window_size
                new_data[new_idx] = self.data[i, ..., start_idx:end_idx]

        new_index_to_info = {
            original_idx * num_windows + window_idx: {
                **info.copy(),
                "window_index": window_idx,
                "original_index": original_idx
            }
            for original_idx, info in self.index_to_info.items()
            for window_idx in range(num_windows)
        }

        return new_data, new_index_to_info

    def validate_split(self) -> Tuple[bool, List[str]]:
        try:
            new_data, new_index_to_info = self.split()
        except Exception as e:
            return False, [f"Split failed: {str(e)}"]

        errors = []
        samples = self.data.shape[0]
        time = self.data.shape[-1]
        num_windows = time // self.window_size

        # Validate data shape
        expected_samples = samples * num_windows
        if new_data.shape[0] != expected_samples:
            errors.append(f"Wrong number of samples: expected {expected_samples}, got {new_data.shape[0]}")

        # Validate data content
        for i in range(samples):
            for j in range(num_windows):
                new_idx = i * num_windows + j
                start_idx = j * self.window_size
                end_idx = start_idx + self.window_size
                original_slice = self.data[i, ..., start_idx:end_idx]
                split_window = new_data[new_idx]

                if not torch.allclose(original_slice, split_window):
                    errors.append(f"Data mismatch at sample {i}, window {j}")

                # Validate index mapping
                if new_idx not in new_index_to_info:
                    errors.append(f"Missing index mapping for {new_idx}")
                elif new_index_to_info[new_idx]["window_index"] != j:
                    errors.append(f"Incorrect window index for {new_idx}")

        return len(errors) == 0, errors


class DataSplitter:
    def __init__(
            self,
            data: torch.Tensor,
            val_split: float = 0.2,
            test_split: float = 0.1,
            seed: Optional[int] = 42,
            stratify_by: Optional[torch.Tensor] = None,
            balanced_split: bool = False
    ):
        if not 0 <= val_split + test_split < 1:
            raise ValueError("Sum of val_split and test_split must be less than 1")

        self.data = data
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.stratify_by = stratify_by
        self.balanced_split = balanced_split

        if stratify_by is not None and balanced_split:
            raise ValueError("Cannot use both stratification and balanced splitting")

        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)

        self._calculate_split_sizes()

    def _calculate_split_sizes(self) -> None:
        total_samples = len(self.data)
        self.test_size = int(total_samples * self.test_split)
        self.val_size = int(total_samples * self.val_split)
        self.train_size = total_samples - self.val_size - self.test_size

    def _print_split_details(self) -> None:
        total = len(self.data)
        print(f"Train: {self.train_size} ({self.train_size / total:.1%})")
        print(f"Val: {self.val_size} ({self.val_size / total:.1%})")
        print(f"Test: {self.test_size} ({self.test_size / total:.1%})")

    def split_data(self) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        if self.balanced_split:
            return self._balanced_split()
        elif self.stratify_by is not None:
            return self._stratified_split()
        else:
            return self._random_split()

    def _random_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = torch.randperm(len(self.data), generator=self.gen).numpy()
        train = np.sort(indices[:self.train_size])
        val = np.sort(indices[self.train_size:self.train_size + self.val_size])
        test = np.sort(indices[self.train_size + self.val_size:])
        self._print_split_details()
        return train, val, test

    def _stratified_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split

        unique_labels = torch.unique(self.stratify_by)
        train_idx, temp_idx = train_test_split(
            np.arange(len(self.data)),
            test_size=self.val_size + self.test_size,
            stratify=self.stratify_by,
            random_state=self.seed
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=self.test_size / (self.val_size + self.test_size),
            stratify=self.stratify_by[temp_idx],
            random_state=self.seed
        )

        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    def _balanced_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stats = self._calculate_stats()
        quantiles = np.linspace(0, 100, num=10 + 1)[:-1]  # 10 quantiles

        indices_by_quantile = self._assign_to_quantiles(stats, quantiles)
        train_idx, val_idx, test_idx = [], [], []

        for q_indices in indices_by_quantile:
            np.random.shuffle(q_indices)
            n = len(q_indices)
            n_test = int(n * self.test_split)
            n_val = int(n * self.val_split)
            n_train = n - n_test - n_val

            train_idx.extend(q_indices[:n_train])
            val_idx.extend(q_indices[n_train:n_train + n_val])
            test_idx.extend(q_indices[n_train + n_val:])

        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    def _calculate_stats(self) -> np.ndarray:
        return self.data.mean(tuple(range(1, self.data.ndim))).numpy()

    def _assign_to_quantiles(self, stats: np.ndarray, quantiles: np.ndarray) -> List[np.ndarray]:
        return [
            np.where(np.logical_and(
                stats >= np.percentile(stats, q),
                stats < np.percentile(stats, q + 10)
            ))[0]
            for q in quantiles
        ]
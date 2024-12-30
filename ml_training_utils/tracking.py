from typing import Dict, Optional, List, Union, Any
import os
import json
import uuid
import numpy as np
import torch
import pandas as pd
from pathlib import Path


class RunTracker:
    """Tracks and manages machine learning experiment runs."""
    def __init__(self, base_dir: Union[str, Path]) -> None:
        """
        Initialize the RunTracker class.

        Args:
        - base_dir (str): The base directory where all runs will be stored.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.index_file = self.base_dir / "runs_index.json"
        self.runs_index = self._load_index()

    def _load_index(self) -> Dict[str, Dict]:
        """Load or create the index file."""
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save the index to a file."""
        with open(self.index_file, "w") as f:
            json.dump(self.runs_index, f, indent=4)

    def create_run(
            self,
            description: str,
            hparams: Dict[str, Any],
            run_id: Optional[str] = None
    ) -> str:
        """
        Create a new run.

        Args:
        - description (str): A short description of the run.
        - hparams (Dict): A dictionary of hyperparameters for the run.
        - run_id (str): Custom run ID. If None, a UUID will be generated.

        Returns:
        - str: The unique ID of the created run.
        """

        run_id = run_id or str(uuid.uuid4())
        run_folder = self.base_dir / run_id

        subfolders = {'plots', 'checkpoints', 'metrics'}
        folders = {name: str(run_folder / name) for name in subfolders}

        run_folder.mkdir(exist_ok=True)
        for folder in subfolders:
            (run_folder / folder).mkdir(exist_ok=True)

        processed_hparams = self._process_hyperparameters(hparams)

        metadata = {
            "description": description,
            "hparams": hparams,
            "run_folder": run_folder,
            **folders,

        }
        self.runs_index[run_id] = metadata

        self._save_index()

        # Save metadata to the run folder
        with open(os.path.join(run_folder, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        return run_id

    def _process_hyperparameters(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        """Process hyperparameters for JSON serialization."""
        if 'loss_fn_hyper_parameters' not in hparams:
            return hparams

        return {
            **hparams,
            'loss_fn_hyper_parameters': {
                key: {
                    **value,
                    'weight': value['weight'].tolist()
                    if isinstance(value['weight'], (np.ndarray, torch.Tensor))
                    else value['weight']
                }
                for key, value in hparams['loss_fn_hyper_parameters'].items()
            }
        }

    def search_runs(self, keyword: Optional[str] = None) -> List[Dict]:
        """
        Search for runs by keyword in the description.

        Args:
        - keyword (str): The keyword to search for in descriptions.

        Returns:
        - List[Dict]: A list of runs matching the keyword.
        """
        if not keyword:
            return list(self.runs_index.values())

        return [
            run for run in self.runs_index.values()
            if keyword.lower() in run["description"].lower()
        ]

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get metadata for a specific run."""
        try:
            return self.runs_index[run_id]
        except KeyError:
            raise KeyError(f"Run ID {run_id} not found")

    def save_text_to_run(self, run_id: str, filename: str, content: str) -> None:
        """Save text content to a run's folder."""
        run_metadata = self.get_run(run_id)
        file_path = Path(run_metadata["run_folder"]) / filename

        with open(file_path, "w") as f:
            f.write(content)

    def _clean_up_index(self) -> None:
        """
        Remove runs from the index if their directories no longer exist.
        """
        keys_to_remove = [
            run_id for run_id, metadata in self.runs_index.items()
            if not os.path.exists(metadata["run_folder"])
        ]
        for run_id in keys_to_remove:
            del self.runs_index[run_id]
        self._save_index()

    def cleanup(self) -> None:
        """Clean up incomplete runs and index."""
        from general import remove_non_complete_runs
        remove_non_complete_runs(self.base_dir)
        self._clean_up_index()

    def gather_metrics(
            self,
            file_name: Optional[str] = None,
            run_ids: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Gather metrics DataFrames from all or specified runs.

        Args:
        - file_name (str, optional): Name of the metric file to look for (e.g., 'train', 'test').
                                     If None, will gather all CSV files.
        - run_ids (List[str], optional): List of specific run IDs to gather metrics from.
                                       If None, will gather from all runs.

        Returns:
        - Dict[str, pd.DataFrame]: Dictionary mapping run IDs to their respective metrics DataFrames
        """
        files_dict = {}
        runs_to_process = run_ids if run_ids is not None else self.runs_index.keys()

        for run_id in runs_to_process:
            run_metadata = self.get_run(run_id)
            if not run_metadata:
                continue

            metrics_folder = run_metadata.get('metrics')
            if not metrics_folder or not os.path.exists(metrics_folder):
                continue

            # Get all CSV files in the metrics folder
            metric_files = [f for f in os.listdir(metrics_folder) if f.endswith('.csv')]

            for metric_file in metric_files:
                # If file_name is specified, only process matching files
                if file_name and not metric_file.startswith(f"{file_name}"):
                    continue

                file_path = os.path.join(metrics_folder, metric_file)
                try:
                    df = pd.read_csv(file_path)
                    files_dict[f"{run_id}_{metric_file[:-4]}"] = df
                except Exception as e:
                    print(f"Error reading metrics file {file_path}: {str(e)}")

        return files_dict

    def process_metrics(
            self,
            metrics_dict: Dict[str, pd.DataFrame],
            columns: Optional[List[str]] = None,
            row_range: Optional[tuple] = None,
            last_row_only: bool = False,
            aggregation: Optional[str] = None,
            filters: Optional[Dict] = None,
            sort_by: Optional[str] = None,
            ascending: bool = True,
            as_df: bool = True
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Process gathered metrics DataFrames with various operations.

        Args:
        - metrics_dict (Dict[str, pd.DataFrame]): Dictionary of metrics DataFrames from gather_metrics
        - columns (List[str], optional): Specific columns to include
        - row_range (tuple, optional): Range of rows to include (start, end)
        - last_row_only (bool): If True, only returns the last row of each DataFrame
        - aggregation (str, optional): Aggregation function to apply ('mean', 'sum', 'max', 'min')
        - filters (Dict, optional): Dictionary of column:value pairs to filter rows
        - sort_by (str, optional): Column name to sort by
        - ascending (bool): Sort order if sort_by is specified
        - as_df (bool): to return a dict or df

        Returns:
        - Dict[str, pd.DataFrame] or pd.DataFrame: Processed metrics DataFrames
        """
        processed = {}

        for run_id, df in metrics_dict.items():
            processed_df = df.copy()

            if columns:
                available_cols = [col for col in columns if col in processed_df.columns]
                processed_df = processed_df[available_cols]

            if row_range and not last_row_only:
                start, end = row_range
                processed_df = processed_df.iloc[start:end]

            if last_row_only:
                processed_df = processed_df.iloc[[-1]]

            if filters:
                for col, value in filters.items():
                    if col not in processed_df.columns:
                        continue
                    if isinstance(value, (list, tuple)):
                        processed_df = processed_df[processed_df[col].isin(value)]
                    else:
                        processed_df = processed_df[processed_df[col] == value]

            if aggregation and not last_row_only:
                agg_map = {
                    'mean': processed_df.mean(),
                    'sum': processed_df.sum(),
                    'max': processed_df.max(),
                    'min': processed_df.min()
                }
                if aggregation.lower() in agg_map:
                    processed_df = pd.DataFrame([agg_map[aggregation.lower()]])

            if sort_by and sort_by in processed_df.columns:
                processed_df = processed_df.sort_values(by=sort_by, ascending=ascending)

            processed[run_id] = processed_df

        if not as_df:
            return processed

        dfs = []
        for key, df in processed.items():
            df = df.copy()
            df['Category'] = key
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def get_metric_summary(self, metrics_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a summary of metrics across all runs.

        Args:
        - metrics_dict (Dict[str, pd.DataFrame]): Dictionary of metrics DataFrames

        Returns:
        - pd.DataFrame: Summary DataFrame with basic statistics for each metric
        """
        summaries = []

        for run_id, df in metrics_dict.items():
            summary = df.describe()
            summary['run_id'] = run_id
            summaries.append(summary)

        return pd.concat(summaries, axis=0)

    def gather_and_display_plots(self,
                                 task_name: str,
                                 run_ids: List[str] = None,
                                 figsize: tuple = (15, 10)) -> None:
        """
        Gather and display plots for a specific task across multiple runs.

        Args:
        - task_name (str): Name of the task (e.g., 'Reconstruction')
        - run_ids (List[str], optional): List of specific run IDs to gather plots from.
                                       If None, will gather from all runs.
        - figsize (tuple): Figure size for the combined plot display
        """
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np

        runs_to_process = run_ids if run_ids is not None else self.runs_index.keys()
        plot_images = []
        run_labels = []

        for run_id in runs_to_process:
            run_metadata = self.get_run(run_id)
            if not run_metadata:
                continue

            plots_folder = run_metadata.get('plots')
            if not plots_folder or not os.path.exists(plots_folder):
                continue

            # Look for matching plot file
            plot_filename = f"{task_name}_metrics.png"
            plot_path = os.path.join(plots_folder, plot_filename)

            if os.path.exists(plot_path):
                try:
                    # Read the image
                    img = Image.open(plot_path)
                    plot_images.append(np.array(img))
                    run_labels.append(run_id)
                except Exception as e:
                    print(f"Error reading plot file {plot_path}: {str(e)}")

        if not plot_images:
            print(f"No plots found for task: {task_name}")
            return

        # Calculate grid dimensions
        n_plots = len(plot_images)
        n_cols = min(3, n_plots)  # Maximum 3 columns
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)

        # Plot each image
        for idx, (img, run_id) in enumerate(zip(plot_images, run_labels)):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Run: {run_id}")
            axes[row, col].axis('off')

        # Turn off any empty subplots
        for idx in range(len(plot_images), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

        return fig
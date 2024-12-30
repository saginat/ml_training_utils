from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod


class BaseMetricsTracker(ABC):
    def __init__(self, metrics_functions: Dict[str, Callable]):
        """
        Initialize BaseMetricsTracker.

        Args:
            metrics_functions: Dictionary mapping metric names to their calculation functions
        """
        self.metrics_functions = metrics_functions
        self.epoch_data = {}
        self.predictions = {}
        self.ground_truth = {}
        self.current_epoch = 1
        self.task_losses = {}
        self.total_losses = []

    def start_epoch(self) -> None:
        """Reset tracking data for new epoch."""
        self.predictions = {}
        self.ground_truth = {}
        self.task_losses = {}
        self.total_losses = []

    @abstractmethod
    def update(self, task_name: str, preds: Optional[torch.Tensor] = None,
               targets: Optional[torch.Tensor] = None, loss: Optional[float] = None) -> None:
        """Update metrics for a task."""
        pass

    def update_total_loss(self, total_loss: float) -> None:
        """Record total loss for current batch."""
        self.total_losses.append(float(total_loss))

    @abstractmethod
    def calculate_epoch_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for all tasks at epoch end."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get all stored metrics."""
        return self.epoch_data

    def dict_to_dataframe(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Convert metrics to DataFrame and optionally save."""
        data = self.get_metrics()
        flattened_data = []

        for task, epochs in data.items():
            for epoch, metrics in epochs.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        flattened_data.append((epoch, f"{task}_{metric}", value))
                else:
                    flattened_data.append((epoch, task, metrics))

        df = pd.DataFrame(flattened_data, columns=["epoch", "metric", "value"])
        result_df = df.pivot(index='epoch', columns='metric', values='value')
        result_df.index.name = 'Epoch'

        if save_path:
            result_df.to_csv(save_path)

        return result_df
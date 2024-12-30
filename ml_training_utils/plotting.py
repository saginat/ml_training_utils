from typing import Dict, Optional, List, Any, Union
from pathlib import Path
import itertools
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class BaseMetricsPlotter:
    def __init__(
            self,
            train_metrics: Dict[str, Dict[int, Any]],
            val_metrics: Optional[Dict[str, Dict[int, Any]]] = None,
            test_metrics: Optional[Dict[str, Dict[int, Any]]] = None,
            baseline_metrics_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.baseline_metrics_dict = baseline_metrics_dict or {}
        self.colors = {
            "train": "blue",
            "validation": "orange",
            "test": "green"
        }
        self.baseline_color = "gray"
        self.baseline_markers = self._generate_baseline_markers()
        self.tasks = [task for task in self.train_metrics.keys() if task != "Total_Loss"]

    def _generate_baseline_markers(self) -> Dict[str, Dict[str, str]]:
        matplotlib_markers = itertools.cycle(["o", "s", "D", "x", "+", "^"])
        plotly_markers = itertools.cycle(["circle", "square", "diamond", "cross", "x"])
        return {
            baseline: {
                "matplotlib": next(matplotlib_markers),
                "plotly": next(plotly_markers)
            }
            for baseline in self.baseline_metrics_dict.keys()
        }

    def _save_figure(self, fig: Union[plt.Figure, go.Figure], save_path: str) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(fig, go.Figure):
            fig.write_html(save_path)
        else:
            fig.savefig(save_path, bbox_inches='tight')

    def _get_dataset_metrics(self, dataset_name: str) -> Dict[str, Dict[int, Any]]:
        metrics_map = {
            'train': self.train_metrics,
            'val': self.val_metrics,
            'test': self.test_metrics,
        }
        if dataset_name not in metrics_map:
            raise ValueError(f"Invalid dataset: {dataset_name}. Use ['train', 'val', 'test']")
        metrics = metrics_map[dataset_name]
        if metrics is None:
            raise ValueError(f"Metrics for {dataset_name} are empty")
        return metrics


class MatplotlibMetricsPlotter(BaseMetricsPlotter):
    def plot_task_metrics(
            self,
            task_name: str,
            save: Optional[str] = None,
            plot: bool = True
    ) -> plt.Figure:
        metrics = list(next(iter(self.train_metrics[task_name].values())).keys()) + ["Total_Loss"]
        num_metrics = len(metrics)
        num_cols = 3
        num_rows = (num_metrics + num_cols - 1) // num_cols  # Ceiling division

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.reshape(-1) if num_metrics > 3 else np.array([axes]).flatten()
        handles, labels = [], []

        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].set_visible(False)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            x_train = list(self.train_metrics[task_name].keys())

            train_values = [
                self.train_metrics["Total_Loss"][epoch] if metric == "Total_Loss"
                else self.train_metrics[task_name][epoch][metric]
                for epoch in x_train
            ]
            train_line, = ax.plot(x_train, train_values, color=self.colors["train"])

            if "Train" not in labels:
                handles.append(train_line)
                labels.append("Train")

            for dataset, color in [("validation", "orange"), ("test", "green")]:
                if getattr(self, f"{dataset}_metrics"):
                    dataset_metrics = getattr(self, f"{dataset}_metrics")
                    x_data = list(dataset_metrics[task_name].keys())
                    values = [
                        dataset_metrics["Total_Loss"][epoch] if metric == "Total_Loss"
                        else dataset_metrics[task_name][epoch][metric]
                        for epoch in x_data
                    ]
                    line, = ax.plot(x_data, values, color=color)
                    if dataset.title() not in labels:
                        handles.append(line)
                        labels.append(dataset.title())

            self._add_baselines(ax, task_name, metric, x_train, handles, labels)
            ax.set_title(f"{metric.capitalize()} for {task_name}")
            ax.grid(True)

        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save:
            self._save_figure(fig, save)
        if plot:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def _add_baselines(self, ax, task_name: str, metric: str, x_data: List[int],
                       handles: List, labels: List) -> None:
        for baseline_name, baseline_metrics in self.baseline_metrics_dict.items():
            if task_name in baseline_metrics.index and metric in baseline_metrics.columns:
                value = baseline_metrics.loc[task_name, metric]
                line, = ax.plot(
                    x_data,
                    [value] * len(x_data),
                    linestyle="dotted",
                    color=self.baseline_color,
                    marker=self.baseline_markers[baseline_name]['matplotlib']
                )
                if baseline_name not in labels:
                    handles.append(line)
                    labels.append(baseline_name)


class PlotlyMetricsPlotter(BaseMetricsPlotter):
    def plot_task_metrics(
            self,
            task_name: str,
            save: Optional[str] = None,
            plot: bool = True
    ) -> go.Figure:
        metrics = list(next(iter(self.train_metrics[task_name].values())).keys()) + ["Total_Loss"]
        num_metrics = len(metrics)
        num_cols = 3
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[f"{metric.capitalize()} for {task_name}" for metric in metrics] +
                           [''] * (num_rows * num_cols - num_metrics)  # Empty titles for unused subplots
        )

        legend_shown = {"train": False, "validation": False, "test": False}

        for idx, metric in enumerate(metrics):
            row, col = divmod(idx, 2)
            row += 1
            col += 1

            x_train = list(self.train_metrics[task_name].keys())
            train_values = [
                self.train_metrics["Total_Loss"][epoch] if metric == "Total_Loss"
                else self.train_metrics[task_name][epoch][metric]
                for epoch in x_train
            ]

            fig.add_trace(
                go.Scatter(
                    x=x_train,
                    y=train_values,
                    mode="lines+markers",
                    name="Train",
                    line=dict(color=self.colors["train"]),
                    legendgroup="train",
                    showlegend=not legend_shown["train"]
                ),
                row=row, col=col
            )
            legend_shown["train"] = True

            self._add_validation_test_traces(fig, task_name, metric, row, col, legend_shown)
            self._add_baseline_traces(fig, task_name, metric, x_train, row, col, legend_shown)

        fig.update_layout(height=800, width=1000, showlegend=True)

        if save:
            self._save_figure(fig, save)
        if plot:
            fig.show()

        return fig

    def _add_validation_test_traces(self, fig: go.Figure, task_name: str, metric: str,
                                    row: int, col: int, legend_shown: Dict[str, bool]) -> None:
        for dataset in ['validation', 'test']:
            dataset_metrics = getattr(self, f"{dataset}_metrics")
            if dataset_metrics:
                x_data = list(dataset_metrics[task_name].keys())
                values = [
                    dataset_metrics["Total_Loss"][epoch] if metric == "Total_Loss"
                    else dataset_metrics[task_name][epoch][metric]
                    for epoch in x_data
                ]

                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=values,
                        mode="lines+markers",
                        name=dataset.title(),
                        line=dict(color=self.colors[dataset]),
                        legendgroup=dataset,
                        showlegend=not legend_shown[dataset]
                    ),
                    row=row, col=col
                )
                legend_shown[dataset] = True

    def _add_baseline_traces(self, fig: go.Figure, task_name: str, metric: str,
                             x_data: List[int], row: int, col: int,
                             legend_shown: Dict[str, bool]) -> None:
        for baseline_name, baseline_metrics in self.baseline_metrics_dict.items():
            if task_name in baseline_metrics.index and metric in baseline_metrics.columns:
                value = baseline_metrics.loc[task_name, metric]
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=[value] * len(x_data),
                        mode="lines+markers",
                        name=baseline_name,
                        line=dict(color=self.baseline_color, dash="dot"),
                        marker=dict(symbol=self.baseline_markers[baseline_name]['plotly']),
                        showlegend=baseline_name not in legend_shown
                    ),
                    row=row, col=col
                )
                legend_shown[baseline_name] = True
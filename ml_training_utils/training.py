from typing import Optional, Dict, Any, Union, Type
import os
import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
from datetime import datetime

class BaseTrainer:
    """
    A base training class providing core functionality for PyTorch model training.

    This class handles:
    - Model training and validation loops
    - Checkpoint management
    - Gradient tracking and clipping
    - Learning rate scheduling
    - Metric tracking
    - Progress monitoring

    Examples:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> train_loader = DataLoader(train_dataset, batch_size=32)
        >>> test_loader = DataLoader(test_dataset, batch_size=32)
        >>> tracker = RunTracker("experiments")
        >>> hyperparams = {
        ...     "num_epochs": 10,
        ...     "device": "cuda",
        ...     "max_norm": 1.0
        ... }
        >>> trainer = BaseTrainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     train_dataloader=train_loader,
        ...     test_dataloader=test_loader,
        ...     tracker=tracker,
        ...     hyperparams=hyperparams
        ... )
        >>> metrics_tr, metrics_test = trainer.train()
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            tracker: Any,
            hyperparams: Dict[str, Any],
            scheduler=None,
            tracker_description: Optional[str] = None,
            run_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the base trainer with core training components.

        Args:
            model (torch.nn.Module): The neural network model to train.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            train_dataloader (DataLoader): DataLoader for training data.
            test_dataloader (DataLoader): DataLoader for test/validation data.
            tracker (object): Tracker for logging and tracking experiments.
            hyperparams (dict): Comprehensive dictionary of experiment hyperparameters.
            scheduler (optional): Learning rate scheduler. Default is None.
            tracker_description (str, optional): Description for the tracker run. Default is None.
            run_id(str, optional), instead of creating a new run id, load a previous one
        """

        """Initialize the BaseTrainer with training components."""
        self._validate_inputs(model, optimizer, train_dataloader, test_dataloader, hyperparams)
        self.default_values_used = {}
        self._initialize_core_components(model, optimizer, train_dataloader,
                                         test_dataloader, tracker, scheduler)

        self._setup_hyperparameters(hyperparams)
        self._calculate_intervals()
        self._initialize_tracking(tracker_description, run_id, hyperparams)
        self._print_default_values()

    def _validate_inputs(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            hyperparams: Dict[str, Any]
    ) -> None:
        """Validate input parameters and their compatibility."""
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a PyTorch Module")

        if not isinstance(optimizer, Optimizer):
            raise TypeError("Optimizer must be a PyTorch Optimizer")

        if not isinstance(train_dataloader, DataLoader):
            raise TypeError("train_dataloader must be a PyTorch DataLoader")

        if not isinstance(test_dataloader, DataLoader):
            raise TypeError("test_dataloader must be a PyTorch DataLoader")

        required_hyperparams = {"num_epochs", "device"}
        missing = required_hyperparams - hyperparams.keys()
        if missing:
            raise ValueError(f"Missing required hyperparameters: {missing}")

    def _initialize_tracking(self, tracker_description: Optional[str], run_id: Optional[str],
                             hyperparams: Dict[str, Any]) -> None:
        """Initialize experiment tracking with proper error handling."""
        self.tracker_description = self._get_param_with_default(hyperparams, 'tracker_description', tracker_description)

        try:
            self.run_id = self._create_new_run() if run_id is None else self._load_or_create_run(run_id)
            self.run_checkpoint_dir = self.tracker.runs_index[self.run_id]["checkpoints"]
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tracking: {str(e)}")

    def _create_new_run(self, run_id: Optional[str] = None) -> str:
        """Create a new tracking run with validation."""
        try:
            run_id = self.tracker.create_run(
                description=self.tracker_description,
                hparams=self.hyperparams,
                run_id=run_id
            )
            if self.verbose >= 1:
                print(f'Created new run: {run_id}')
            return run_id
        except Exception as e:
            raise RuntimeError(f"Failed to create new run: {str(e)}")

    def _load_or_create_run(self, run_id: str) -> str:
        """Load existing run or create new one with proper validation."""
        try:
            self.tracker.get_run(run_id)
            if self.verbose >= 1:
                print('Loading existing run...')
            return run_id
        except KeyError:
            if self.verbose >= 1:
                print('Run not found, creating new...')
            return self._create_new_run(run_id)

    def _get_param_with_default(self, params_dict, key, default_value):
        """Helper method to get parameter and track if default was used."""
        value = params_dict.get(key, default_value)
        if key not in params_dict:
            self.default_values_used[key] = default_value
        return value

    def _initialize_core_components(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        tracker: Any,
        scheduler
    ) -> None:
        """Initialize core training components."""
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.tracker = tracker
        self.scheduler = scheduler
        self.current_epoch = 1
        self.grad_norms: list = []
        self.grad_norms_clipped: list = []

    def _setup_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Extract and set up training hyperparameters with validation."""
        self.hyperparams = hyperparams
        self.num_epochs = self._get_param_with_default(hyperparams, 'num_epochs', 1)
        self.device = self._get_param_with_default(hyperparams, 'device', 'cuda')
        self.max_norm = self._get_param_with_default(hyperparams, 'max_norm', None)
        self.track_grad = self._get_param_with_default(hyperparams, 'track_grad', False)
        self.verbose = self._get_param_with_default(hyperparams, 'verbose', 1)
        self.best_metric_name = self._get_param_with_default(hyperparams, 'best_metric_name', None)

        # Initialize best-metric-tracking
        self.best_metric_value = self._initialize_best_metric_value()
        self.best_epoch = 1
        self.is_maximization_metric = self.best_metric_value == float('-inf')

    def _calculate_intervals(self) -> None:
        """Calculate intervals for printing, plotting, and checkpointing with validation."""
        print_percentage = self._get_param_with_default(self.hyperparams, 'print_percentage', 0.1)
        plot_percentage = self._get_param_with_default(self.hyperparams, 'plot_percentage', None)
        checkpoints_percentage = self._get_param_with_default(self.hyperparams, 'checkpoints_percentage', None)

        if not 0 < print_percentage <= 1:
            raise ValueError("print_percentage must be between 0 and 1")
        if plot_percentage is not None and not 0 < plot_percentage <= 1:
            raise ValueError("plot_percentage must be between 0 and 1")
        if checkpoints_percentage is not None and not 0 < checkpoints_percentage <= 1:
            raise ValueError("checkpoints_percentage must be between 0 and 1")

        total_prints = round(self.num_epochs * print_percentage)
        total_plots = round(self.num_epochs * plot_percentage) if plot_percentage else 0
        total_checkpoints = round(self.num_epochs * checkpoints_percentage) if checkpoints_percentage else 1

        self.print_interval = max(1, self.num_epochs // total_prints) if total_prints > 0 else self.num_epochs
        self.plot_interval = max(1, self.num_epochs // total_plots) if total_plots > 0 else None
        self.checkpoint_interval = max(1, self.num_epochs // total_checkpoints) if total_checkpoints > 0 else self.num_epochs

    def _print_default_values(self) -> None:
        """Print hyperparameters using default values."""
        if self.default_values_used and self.verbose >= 1:
            print("\nUsing default values for:")
            for param, default_value in self.default_values_used.items():
                print(f"  - {param}: {default_value}")

    def _initialize_best_metric_value(self):
        """
        Initialize the best metric value based on the metric's nature.

        Returns:
            float: Initial best metric value
        """
        if not self.best_metric_name:
            return None

        # Metrics that should be maximized
        maximization_keywords = ['accuracy', 'score', 'f1', 'precision', 'recall']

        # Metrics that should be minimized
        minimization_keywords = ['loss', 'error', 'mse', 'mae', 'rmse']

        # Check if the metric name contains any maximization keywords
        is_maximization_metric = any(
            keyword in self.best_metric_name.lower()
            for keyword in maximization_keywords
        )

        # Check if the metric name contains any minimization keywords
        is_minimization_metric = any(
            keyword in self.best_metric_name.lower()
            for keyword in minimization_keywords
        )

        # Determine the initial value
        if is_maximization_metric:
            return float('-inf')
        elif is_minimization_metric:
            return float('inf')
        else:
            # If we can't determine, default to minimization
            print(
                f"Warning: Could not determine optimization direction for metric {self.best_metric_name}. Defaulting to minimization.")
            return float('inf')

    def _set_loops(self):
        self.epoch_loop = (
            tqdm(range(self.current_epoch, self.num_epochs + 1), desc="epoch loop")
            if self.verbose >= 1
            else range(self.current_epoch, self.num_epochs + 1)
        )
        self.train_loop = (
            tqdm(self.train_dataloader, colour='#1167b1', desc='train dataloader loop')
            if self.verbose == 2
            else self.train_dataloader
        )
        self.test_loop = (
            tqdm(self.test_dataloader, colour='red', desc='test dataloader loop')
            if self.verbose == 2
            else self.test_dataloader
        )
        return None

    def _save_checkpoint(self, epoch: int) -> str:
        """
        Save a training checkpoint.

        Returns:
            str: Path to the saved checkpoint
        """
        checkpoint_path = Path(self.run_checkpoint_dir) / f'checkpoint_epoch_{epoch:04d}.pth'

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'grad_norms': self.grad_norms,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }

        torch.save(checkpoint, checkpoint_path)
        if self.verbose >= 1:
            print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(
            self,
            checkpoint_path: Optional[str] = None,
            epoch: Optional[int] = None
    ) -> Dict[str, Any]:
        """Load a training checkpoint."""
        if not checkpoint_path and not epoch:
            checkpoints = sorted(Path(self.run_checkpoint_dir).glob('checkpoint_epoch_*.pth'))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.run_checkpoint_dir}")
            checkpoint_path = str(checkpoints[-1])
        elif epoch:
            checkpoint_path = str(Path(self.run_checkpoint_dir) / f'checkpoint_epoch_{epoch:04d}.pth')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.grad_norms = checkpoint.get('grad_norms', [])

        if self.verbose >= 1:
            print(f"Checkpoint loaded: {checkpoint_path}")

        return checkpoint

    def _get_grad_norm(self, norm: int = 2) -> float:
        """Calculate gradient norm of model parameters."""
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm) for p in parameters]),
            norm
        )
        return total_norm.item()

    def _handle_batch(self, *args, **kwargs) -> None:
        """Process a single batch of data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _handle_batch")

    def train(self):
        """Execute main training loop with proper setup and error handling."""
        try:
            self._set_loops()

            for epoch in self.epoch_loop:
                if self.verbose >= 1:
                    self.epoch_loop.set_postfix({"lr": self.optimizer.param_groups[0]['lr']})

                self.model.train()
                self._train_epoch(self.train_loop, epoch)

                if self.checkpoint_interval and epoch % self.checkpoint_interval == 0:
                    self._save_checkpoint(epoch)

                torch.cuda.empty_cache()
                self.model.eval()
                self._validate_epoch(self.test_loop, epoch)

            self._save_checkpoint(epoch)
            return (self.metrics_tracker_tr if hasattr(self, 'metrics_tracker_tr') else None,
                    self.metrics_tracker_test if hasattr(self, 'metrics_tracker_test') else None)

        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            self._save_checkpoint(epoch)
            raise

    def continue_training(self, checkpoint_path: Optional[str] = None, epoch: Optional[int] = None, new_total_epochs: Optional[int] = None) -> None:
        """
        Continue training from the last saved checkpoint.

        Args:
            new_total_epochs (int, optional): New total number of epochs to train.
                                              If None, uses the original num_epochs.
            checkpoint_path (str, optional): Specific checkpoint file to load
            epoch (int, optional): Epoch number to load (will find the corresponding checkpoint)
        """
        self.load_checkpoint(checkpoint_path=checkpoint_path, epoch=epoch)
        print(f"Resuming from epoch {self.current_epoch}")

        original_epochs = self.num_epochs
        self.num_epochs = new_total_epochs or self.num_epochs

        try:
            self.epoch_loop = (
                tqdm(range(self.current_epoch + 1, self.num_epochs + 1), desc="epoch loop")
                if self.verbose >= 1
                else range(self.current_epoch + 1, self.num_epochs + 1)
            )
            self.train()
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            self._save_checkpoint(self.current_epoch)
            raise
        finally:
            self.num_epochs = original_epochs

    def _update_best_checkpoint(self, epoch: int, current_metric_value: float) -> bool:
        """Update best checkpoint if metric improves."""
        if not self.best_metric_name:
            return False

        is_better = (current_metric_value > self.best_metric_value
                     if self.is_maximization_metric
                     else current_metric_value < self.best_metric_value)

        if is_better:
            if hasattr(self, 'best_checkpoint_path') and os.path.exists(self.best_checkpoint_path):
                os.remove(self.best_checkpoint_path)

            checkpoint_path = Path(
                self.tracker.runs_index[self.run_id]["checkpoints"]) / f'best_checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metric_value': current_metric_value
            }, checkpoint_path)

            self.best_metric_value = current_metric_value
            self.best_epoch = epoch
            self.best_checkpoint_path = str(checkpoint_path)
            return True
        return False

    def _train_epoch(self, *args, **kwargs) -> None:
        """Run one epoch of training. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _train_epoch")

    def _validate_epoch(self, *args, **kwargs) -> None:
        """Run one epoch of validation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _validate_epoch")
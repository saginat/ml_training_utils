# ML Training Utils

A comprehensive PyTorch-based toolkit for managing machine learning experiments. Streamlines experiment tracking, metrics monitoring, and training workflows.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/ml-training-utils.svg)](https://badge.fury.io/py/ml-training-utils)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

- **Experiment Management**
  ```python
  tracker = RunTracker("experiments")
  run_id = tracker.create_run(description="Training run", hparams={"lr": 0.001})
  ```
  - Automatic experiment tracking and metadata storage
  - Checkpoint management with best model saving


- **Training Pipeline**
  ```python
  trainer = BaseTrainer(
      model=model,
      optimizer=optimizer,
      train_dataloader=train_loader,
      test_dataloader=test_loader,
      tracker=tracker,
      hyperparams={"num_epochs": 10, "device": "cuda"}
  )
  metrics = trainer.train()
  ```
  - Flexible base trainer with customizable training loops
  - Progress tracking with configurable intervals
  - Gradient monitoring and clipping
  - Learning rate scheduling

- **Time Series Processing**
  ```python
  splitter = TimeWindowSplitter(data, index_to_info, window_size=24)
  windowed_data, new_info = splitter.split()
  ```
  - Window-based sequence splitting
  - Dataset partitioning with stratification
  - Balanced splitting for imbalanced data

- **Visualization**
  ```python
  plotter = MatplotlibMetricsPlotter(train_metrics, val_metrics)
  plotter.plot_task_metrics("classification", save="metrics.png")
  ```
  - Real-time training metrics visualization
  - Support for both Matplotlib and Plotly
  - Customizable plotting options

## Installation

```bash
pip install ml-training-utils
```

For development:
```bash
git clone https://github.com/yourusername/ml-training-utils
cd ml-training-utils
pip install -e ".[dev]"
```

## Quick Example

```python
from ml_training_utils import RunTracker, BaseTrainer
import torch.nn as nn

# Initialize tracking
tracker = RunTracker("experiments")

# Define model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

# Create trainer
trainer = BaseTrainer(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    tracker=tracker,
    hyperparams={
        "num_epochs": 10,
        "device": "cuda",
        "max_norm": 1.0,
        "print_percentage": 0.1
    }
)

# Train with automatic tracking
metrics_train, metrics_test = trainer.train()
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ml_training_utils,
  author = {Sagi Nathan},
  title = {ML Training Utils: A PyTorch Experiment Management Toolkit},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/saginat/ml-training-utils}
}
```
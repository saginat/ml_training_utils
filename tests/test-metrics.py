import pytest
import torch
from ml_training_utils.metrics import BaseMetricsTracker

class SimpleMetricsTracker(BaseMetricsTracker):
    def update(self, task_name, preds=None, targets=None, loss=None):
        if preds is not None and targets is not None:
            if task_name not in self.predictions:
                self.predictions[task_name] = []
                self.ground_truth[task_name] = []
            self.predictions[task_name].append(preds)
            self.ground_truth[task_name].append(targets)
        
        if loss is not None:
            if task_name not in self.task_losses:
                self.task_losses[task_name] = []
            self.task_losses[task_name].append(loss)

    def calculate_epoch_metrics(self):
        metrics = {}
        for task_name in self.predictions:
            metrics[task_name] = {}
            for metric_name, metric_fn in self.metrics_functions.items():
                preds = torch.cat(self.predictions[task_name])
                targets = torch.cat(self.ground_truth[task_name])
                metrics[task_name][metric_name] = metric_fn(preds, targets)
        return metrics

def dummy_accuracy(preds, targets):
    return torch.mean((preds > 0.5).float() == targets.float()).item()

@pytest.fixture
def tracker():
    return SimpleMetricsTracker({'accuracy': dummy_accuracy})

def test_metrics_tracker_initialization(tracker):
    assert isinstance(tracker.metrics_functions, dict)
    assert 'accuracy' in tracker.metrics_functions
    assert tracker.current_epoch == 1

def test_metrics_tracker_update(tracker):
    preds = torch.tensor([0.7, 0.3, 0.8])
    targets = torch.tensor([1.0, 0.0, 1.0])
    
    tracker.update('classification', preds, targets, loss=0.5)
    
    assert 'classification' in tracker.predictions
    assert len(tracker.predictions['classification']) == 1
    assert len(tracker.task_losses['classification']) == 1

def test_metrics_tracker_epoch_calculation(tracker):
    preds = torch.tensor([0.7, 0.3, 0.8])
    targets = torch.tensor([1.0, 0.0, 1.0])
    
    tracker.update('classification', preds, targets, loss=0.5)
    metrics = tracker.calculate_epoch_metrics()
    
    assert 'classification' in metrics
    assert 'accuracy' in metrics['classification']
    assert isinstance(metrics['classification']['accuracy'], float)

def test_metrics_tracker_reset(tracker):
    preds = torch.tensor([0.7, 0.3, 0.8])
    targets = torch.tensor([1.0, 0.0, 1.0])
    
    tracker.update('classification', preds, targets, loss=0.5)
    tracker.start_epoch()
    
    assert not tracker.predictions
    assert not tracker.ground_truth
    assert not tracker.task_losses
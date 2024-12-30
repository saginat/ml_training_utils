import pytest
import os
import json
import shutil
from pathlib import Path
from ml_training_utils.tracking import RunTracker

@pytest.fixture
def tracker(tmp_path):
    return RunTracker(str(tmp_path))

def test_run_creation(tracker):
    run_id = tracker.create_run(
        description="Test run",
        hparams={'learning_rate': 0.01}
    )
    
    assert run_id in tracker.runs_index
    assert os.path.exists(os.path.join(tracker.base_dir, run_id))
    
    # Check folders were created
    run_path = Path(tracker.runs_index[run_id]['run_folder'])
    assert (run_path / 'plots').exists()
    assert (run_path / 'checkpoints').exists()
    assert (run_path / 'metrics').exists()

def test_run_metadata(tracker):
    hparams = {
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100
    }
    
    run_id = tracker.create_run(
        description="Test run",
        hparams=hparams
    )
    
    run_info = tracker.get_run(run_id)
    assert run_info['description'] == "Test run"
    assert run_info['hparams'] == hparams

def test_search_runs(tracker):
    tracker.create_run(description="Training run 1", hparams={})
    tracker.create_run(description="Training run 2", hparams={})
    tracker.create_run(description="Validation run", hparams={})
    
    training_runs = tracker.search_runs(keyword="Training")
    assert len(training_runs) == 2
    
    all_runs = tracker.search_runs()
    assert len(all_runs) == 3

def test_cleanup(tracker):
    run_id = tracker.create_run(description="Test run", hparams={})
    run_path = Path(tracker.runs_index[run_id]['run_folder'])
    
    # Remove run folder to simulate incomplete run
    shutil.rmtree(run_path)
    
    tracker.cleanup()
    assert run_id not in tracker.runs_index

def test_invalid_run_id(tracker):
    with pytest.raises(KeyError):
        tracker.get_run("nonexistent_run")

def test_metrics_gathering(tracker):
    # Create a run with some metrics
    run_id = tracker.create_run(description="Test run", hparams={})
    metrics_path = Path(tracker.runs_index[run_id]['metrics'])
    
    # Create dummy metrics file
    metrics_data = "epoch,loss,accuracy\n1,0.5,0.8\n2,0.3,0.9\n"
    with open(metrics_path / "train_metrics.csv", "w") as f:
        f.write(metrics_data)
    
    metrics_dict = tracker.gather_metrics()
    assert len(metrics_dict) == 1
    assert f"{run_id}_train_metrics" in metrics_dict
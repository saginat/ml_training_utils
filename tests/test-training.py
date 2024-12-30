import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ml_training_utils.training import BaseTrainer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

class SimpleTrainer(BaseTrainer):
    def _handle_batch(self, batch):
        x, y = batch
        y_pred = self.model(x)
        loss = nn.MSELoss()(y_pred, y)
        return loss
        
    def _train_epoch(self, train_loop, epoch):
        for batch in train_loop:
            self.optimizer.zero_grad()
            loss = self._handle_batch(batch)
            loss.backward()
            if self.max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            
    def _validate_epoch(self, test_loop, epoch):
        with torch.no_grad():
            for batch in test_loop:
                loss = self._handle_batch(batch)

@pytest.fixture
def trainer(tmp_path):
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32)
    test_loader = DataLoader(dataset, batch_size=32)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    return SimpleTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        tracker=None,
        hyperparams={
            "num_epochs": 2,
            "device": "cpu",
            "max_norm": 1.0
        }
    )

def test_trainer_initialization(trainer):
    assert trainer.num_epochs == 2
    assert trainer.device == "cpu"
    assert trainer.max_norm == 1.0
    assert trainer.current_epoch == 1

def test_training_loop(trainer):
    trainer.train()
    assert trainer.current_epoch == trainer.num_epochs

def test_checkpoint_saving(trainer, tmp_path):
    # Set up checkpoint directory
    trainer.run_checkpoint_dir = str(tmp_path)
    
    # Save checkpoint
    checkpoint_path = trainer._save_checkpoint(1)
    assert os.path.exists(checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint['epoch'] == 1
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint

def test_gradient_tracking(trainer):
    trainer.track_grad = True
    
    # Run one epoch
    for batch in trainer.train_dataloader:
        trainer.optimizer.zero_grad()
        loss = trainer._handle_batch(batch)
        loss.backward()
        if trainer.max_norm:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.max_norm)
            grad_norm = trainer._get_grad_norm()
            assert grad_norm <= trainer.max_norm
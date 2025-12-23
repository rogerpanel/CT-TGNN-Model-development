"""
Training Script for CT-TGNN

Handles:
- Model training with multiple optimizers
- Checkpoint saving/loading
- TensorBoard logging
- Evaluation metrics computation

Author: Roger Nick Anaedevha
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import yaml
from tqdm import tqdm
from typing import Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ct_tgnn import CTTGNN
from models.baselines.strgnn import StrGNN
from models.baselines.cnn_lstm import CNNLSTM
from data.data_loaders import get_dataloader


class Trainer:
    """Main trainer class for CT-TGNN and baselines."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.model = self._build_model()
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs']
        )

        # Logging
        self.writer = SummaryWriter(log_dir=config['logging']['log_dir'])

        # Checkpointing
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Metrics
        self.best_val_acc = 0.0
        self.global_step = 0

    def _build_model(self):
        """Build model based on config."""
        model_name = self.config['model']['name']

        if model_name == 'CT-TGNN':
            model = CTTGNN(
                node_feat_dim=self.config['model']['node_feat_dim'],
                edge_feat_dim=self.config['model']['edge_feat_dim'],
                hidden_dim=self.config['model']['hidden_dim'],
                num_ode_blocks=self.config['model']['num_ode_blocks'],
                num_classes=self.config['model'].get('num_classes', 2)
            )
        elif model_name == 'StrGNN':
            model = StrGNN(
                node_feat_dim=self.config['model']['node_feat_dim'],
                hidden_dim=self.config['model']['hidden_dim']
            )
        elif model_name == 'CNN-LSTM':
            model = CNNLSTM(
                input_dim=self.config['model']['edge_feat_dim'],
                hidden_dim=self.config['model']['hidden_dim']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')

        for batch in pbar:
            # Move to device
            if hasattr(batch, 'to'):
                batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if hasattr(batch, 'x'):  # Graph data
                outputs = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.timestamps,
                    batch.batch
                )
                loss = nn.CrossEntropyLoss()(outputs['logits'], batch.y)
                _, predicted = outputs['logits'].max(1)
                correct += predicted.eq(batch.y).sum().item()
                total += batch.y.size(0)

            else:  # Sequential data
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(features)
                loss = nn.CrossEntropyLoss()(outputs['logits'], labels)
                _, predicted = outputs['logits'].max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Logging
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total if total > 0 else 0.0
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                if hasattr(batch, 'to'):
                    batch = batch.to(self.device)

                if hasattr(batch, 'x'):
                    outputs = self.model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.timestamps,
                        batch.batch
                    )
                    loss = nn.CrossEntropyLoss()(outputs['logits'], batch.y)
                    _, predicted = outputs['logits'].max(1)
                    correct += predicted.eq(batch.y).sum().item()
                    total += batch.y.size(0)
                else:
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(features)
                    loss = nn.CrossEntropyLoss()(outputs['logits'], labels)
                    _, predicted = outputs['logits'].max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

                total_loss += loss.item()

        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100.0 * correct / total if total > 0 else 0.0
        }

        return metrics

    def train(self):
        """Main training loop."""
        # Data loaders
        train_loader = get_dataloader(
            self.config['data']['dataset'],
            self.config['data']['data_dir'],
            split='train',
            batch_size=self.config['training']['batch_size']
        )

        val_loader = get_dataloader(
            self.config['data']['dataset'],
            self.config['data']['data_dir'],
            split='val',
            batch_size=self.config['training']['batch_size']
        )

        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")

            # Validate
            val_metrics = self.validate(val_loader)
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

            # Logging
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, is_best=True)

            # LR scheduling
            self.scheduler.step()

        print(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.2f}%")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, f"{self.config['model']['name']}_best.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"{self.config['model']['name']}_epoch{epoch}.pt")

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None, help='Override dataset')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config['data']['dataset'] = args.dataset

    # Set random seed
    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get('seed', 42))

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

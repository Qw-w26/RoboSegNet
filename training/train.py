"""
Training script for segment boundary prediction model.

This script trains an MLP to predict segment boundaries from multi-modal features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import wandb
import cv2
import matplotlib.pyplot as plt

from data import AgiBotWorldDataset
from training.dataset_generator import SegmentBoundaryDataset, create_dataloaders
from model import BoundaryPredictionMLP, FocalLoss, create_model


class Trainer:
    """Trainer for boundary prediction model."""

    def __init__(
        self,
        model: BoundaryPredictionMLP,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: Path = Path('./checkpoints'),
        log_dir: Path = Path('./logs'),
        use_wandb: bool = True,
        log_visualizations: bool = True,
        viz_frequency: int = 100
    ):
        """
        Initialize trainer.

        Args:
            model: Boundary prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            use_wandb: Whether to use Weights & Biases for logging
            log_visualizations: Whether to log visualizations (inputs, outputs, videos)
            viz_frequency: Log visualizations every N steps
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Wandb settings
        self.use_wandb = use_wandb
        self.log_visualizations = log_visualizations
        self.viz_frequency = viz_frequency

        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.global_step = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move batch to device
            tokenized_actions = batch['tokenized_actions'].to(self.device)
            proprio_stats = batch['proprio_stats'].to(self.device)
            dino_features = batch['dino_features'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(tokenized_actions, proprio_stats, dino_features)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to tensorboard
            self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/loss_step': loss.item(),
                    'train/step': self.global_step
                }, step=self.global_step)

                # Log visualizations periodically
                if self.log_visualizations and self.global_step % self.viz_frequency == 0:
                    self._log_visualizations(batch, logits, labels)

            self.global_step += 1

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Convert soft labels to hard binary labels (threshold at 0.5)
        all_labels_binary = (all_labels >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_binary, all_preds, average='binary', zero_division=0
        )
        accuracy = (all_preds == all_labels_binary).mean()

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                tokenized_actions = batch['tokenized_actions'].to(self.device)
                proprio_stats = batch['proprio_stats'].to(self.device)
                dino_features = batch['dino_features'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                logits = self.model(tokenized_actions, proprio_stats, dino_features)
                loss = self.criterion(logits, labels)

                # Record
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Convert soft labels to hard binary labels (threshold at 0.5)
        all_labels_binary = (all_labels >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_binary, all_preds, average='binary', zero_division=0
        )
        accuracy = (all_preds == all_labels_binary).mean()

        # ROC AUC (can use soft labels since it works with probabilities)
        try:
            roc_auc = roc_auc_score(all_labels_binary, all_probs)
        except:
            roc_auc = 0.0

        # Confusion matrix
        cm = confusion_matrix(all_labels_binary, all_preds)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }

        return metrics

    def _log_visualizations(self, batch: Dict[str, torch.Tensor], logits: torch.Tensor, labels: torch.Tensor):
        """
        Log visualizations to wandb including inputs, outputs, DINO features, and videos.

        Args:
            batch: Batch of data from dataloader
            logits: Model predictions (logits)
            labels: Ground truth labels
        """
        if not self.use_wandb:
            return

        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of boundary class
        preds = torch.argmax(logits, dim=1)

        # Select a few samples to visualize (max 4)
        num_samples = min(4, len(labels))

        viz_data = {}

        for i in range(num_samples):
            sample_idx = i

            # 1. Log DINO features as heatmap
            dino_feat = batch['dino_features'][sample_idx].cpu().numpy()  # Shape: (3072,) = 3 frames * 1024
            dino_feat_reshaped = dino_feat.reshape(3, 1024)  # 3 frames, 1024 features each

            # Visualize DINO features as image
            fig, ax = plt.subplots(figsize=(10, 3))
            im = ax.imshow(dino_feat_reshaped, aspect='auto', cmap='viridis')
            ax.set_xlabel('Feature Dimension')
            ax.set_ylabel('Frame (begin/middle/end)')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Begin', 'Middle', 'End'])
            plt.colorbar(im, ax=ax)
            plt.title(f'DINO Features - Sample {i}')
            plt.tight_layout()
            viz_data[f'dino_features/sample_{i}'] = wandb.Image(fig)
            plt.close(fig)

            # 2. Log proprio stats as histogram
            proprio_stats = batch['proprio_stats'][sample_idx].cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(proprio_stats)), proprio_stats)
            ax.set_xlabel('Proprio Stat Index')
            ax.set_ylabel('Value')
            ax.set_title(f'Proprioception Statistics - Sample {i}')
            plt.tight_layout()
            viz_data[f'proprio_stats/sample_{i}'] = wandb.Image(fig)
            plt.close(fig)

            # 3. Log tokenized actions (first 50 tokens)
            tokens = batch['tokenized_actions'][sample_idx].cpu().numpy()[:50]
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.bar(range(len(tokens)), tokens)
            ax.set_xlabel('Token Position')
            ax.set_ylabel('Token ID')
            ax.set_title(f'Tokenized Actions (first 50) - Sample {i}')
            plt.tight_layout()
            viz_data[f'tokenized_actions/sample_{i}'] = wandb.Image(fig)
            plt.close(fig)

            # 4. Log prediction info
            label_value = labels[sample_idx].item()
            label_binary = int(label_value >= 0.5)  # Convert soft label to binary
            viz_data[f'predictions/sample_{i}_label'] = label_value  # Log soft label value
            viz_data[f'predictions/sample_{i}_label_binary'] = label_binary  # Log binary label
            viz_data[f'predictions/sample_{i}_pred'] = preds[sample_idx].item()
            viz_data[f'predictions/sample_{i}_prob'] = probs[sample_idx].item()
            viz_data[f'predictions/sample_{i}_correct'] = (preds[sample_idx].item() == label_binary)

        # Log all visualizations
        wandb.log(viz_data, step=self.global_step)

    def _render_predictions_on_video(
        self,
        video_path: str,
        frame_indices: list,
        predictions: list,
        labels: list,
        probabilities: list,
        output_path: str = None
    ):
        """
        Render predictions on video frames.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to render
            predictions: List of predicted labels (0 or 1)
            labels: List of ground truth labels
            probabilities: List of prediction probabilities
            output_path: Path to save rendered video

        Returns:
            Path to rendered video or None if failed
        """
        if not Path(video_path).exists():
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        if output_path is None:
            output_path = f'/tmp/viz_video_{self.global_step}.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        pred_dict = {frame_indices[i]: (predictions[i], labels[i], probabilities[i])
                     for i in range(len(frame_indices))}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # If this frame has a prediction, annotate it
            if frame_idx in pred_dict:
                pred, label, prob = pred_dict[frame_idx]

                # Color: green if correct, red if incorrect
                color = (0, 255, 0) if pred == label else (0, 0, 255)

                # Draw prediction info
                text = f'Pred: {"Boundary" if pred else "Non-boundary"} ({prob:.2f})'
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, color, 2)

                text2 = f'True: {"Boundary" if label else "Non-boundary"}'
                cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)

                # Draw circle to indicate this frame
                cv2.circle(frame, (width - 30, 30), 20, color, -1)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        return output_path

    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Log training metrics
            self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('train/precision', train_metrics['precision'], epoch)
            self.writer.add_scalar('train/recall', train_metrics['recall'], epoch)
            self.writer.add_scalar('train/f1', train_metrics['f1'], epoch)

            if self.use_wandb:
                wandb.log({
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'train/epoch_precision': train_metrics['precision'],
                    'train/epoch_recall': train_metrics['recall'],
                    'train/epoch_f1': train_metrics['f1'],
                    'epoch': epoch
                }, step=self.global_step)

            print(f"\nEpoch {epoch} Train Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  Precision: {train_metrics['precision']:.4f}")
            print(f"  Recall: {train_metrics['recall']:.4f}")
            print(f"  F1: {train_metrics['f1']:.4f}")

            # Validate
            if self.val_loader:
                val_metrics = self.validate()

                # Log validation metrics
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('val/precision', val_metrics['precision'], epoch)
                self.writer.add_scalar('val/recall', val_metrics['recall'], epoch)
                self.writer.add_scalar('val/f1', val_metrics['f1'], epoch)
                self.writer.add_scalar('val/roc_auc', val_metrics['roc_auc'], epoch)

                if self.use_wandb:
                    # Log confusion matrix as wandb table
                    cm = np.array(val_metrics['confusion_matrix'])
                    wandb.log({
                        'val/epoch_loss': val_metrics['loss'],
                        'val/epoch_accuracy': val_metrics['accuracy'],
                        'val/epoch_precision': val_metrics['precision'],
                        'val/epoch_recall': val_metrics['recall'],
                        'val/epoch_f1': val_metrics['f1'],
                        'val/epoch_roc_auc': val_metrics['roc_auc'],
                        'val/confusion_matrix': wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=None,
                            preds=None,
                            class_names=['non-boundary', 'boundary']
                        ) if cm.size > 0 else None,
                        'epoch': epoch
                    }, step=self.global_step)

                print(f"\nEpoch {epoch} Val Metrics:")
                print(f"  Loss: {val_metrics['loss']:.4f}")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Precision: {val_metrics['precision']:.4f}")
                print(f"  Recall: {val_metrics['recall']:.4f}")
                print(f"  F1: {val_metrics['f1']:.4f}")
                print(f"  ROC AUC: {val_metrics['roc_auc']:.4f}")

                # Save best model
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best F1! Saved checkpoint.")

            # Learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'] if self.val_loader else train_metrics['f1'])
                else:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', current_lr, epoch)
                if self.use_wandb:
                    wandb.log({'train/learning_rate': current_lr}, step=self.global_step)
                print(f"  Learning rate: {current_lr:.6f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        self.writer.close()
        print("\nTraining complete!")

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'global_step': self.global_step
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.global_step = checkpoint['global_step']

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {self.current_epoch}, Best Val F1: {self.best_val_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train boundary prediction model')

    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to AgiBotWorld dataset')
    parser.add_argument('--task_ids', type=int, nargs='+', default=[327],
                       help='Task IDs to use for training')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Maximum number of trajectories to use')
    parser.add_argument('--window_size', type=int, default=15,
                       help='Sliding window size')
    parser.add_argument('--stride', type=int, default=1,
                       help='Sliding window stride')
    parser.add_argument('--boundary_tolerance', type=int, default=2,
                       help='Tolerance for boundary labeling')

    # Model arguments
    parser.add_argument('--max_token_length', type=int, default=600,
                       help='Maximum token length for Fast-DROID')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss instead of cross-entropy')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./training_output',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    # Load dataset
    print(f"Loading dataset from {args.data_root}...")
    dataset = AgiBotWorldDataset(
        data_root=Path(args.data_root),
        task_ids=args.task_ids,
        max_trajectories=args.max_trajectories
    )
    dataset.print_statistics()

    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = SegmentBoundaryDataset(
        dataset=dataset,
        window_size=args.window_size,
        stride=args.stride,
        max_token_length=args.max_token_length,
        boundary_tolerance=args.boundary_tolerance,
        device=args.device
    )

    # Create dataloaders
    train_loader, _ = create_dataloaders(
        train_dataset,
        val_dataset=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    print("\nCreating model...")
    model = create_model(
        vocab_size=32000,
        max_token_length=args.max_token_length,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        device=args.device
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Cross-Entropy Loss")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        output_dir=checkpoint_dir,
        log_dir=log_dir
    )

    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved configuration to {output_dir / 'config.json'}")

    # Train
    trainer.train(num_epochs=args.num_epochs, save_every=args.save_every)


if __name__ == "__main__":
    main()

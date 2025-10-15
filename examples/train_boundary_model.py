"""
Example script for training the boundary prediction model.

This script demonstrates how to:
1. Load the AgiBotWorld dataset
2. Generate training data with DINO features and tokenized proprio
3. Train an MLP model to predict segment boundaries
"""

from pathlib import Path
import torch

from data import AgiBotHFDirectDataset
from training import SegmentBoundaryDataset, create_dataloaders, Trainer
from model import create_model, FocalLoss


def main():
    # Configuration
    cache_dir = Path("/home/syx/ucsf/task_seg/agibot_full_data")
    task_ids = [327]
    max_trajectories = 10  # Start with fewer for testing

    window_size = 15
    stride = 1
    max_token_length = 600
    boundary_tolerance = 2

    batch_size = 8  # Smaller batch size for testing
    num_epochs = 2  # Just 2 epochs for initial testing
    learning_rate = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path("./training_output")

    print("="*70)
    print("Boundary Prediction Model Training")
    print("="*70)

    # Step 1: Load dataset
    print(f"\nStep 1: Loading AgiBotWorld dataset from HuggingFace...")
    dataset = AgiBotHFDirectDataset(
        task_ids=task_ids,
        max_trajectories=max_trajectories,
        cache_dir=str(cache_dir),
        download_videos=True,  # Need videos for DINO
        download_proprio=True  # Need H5 for proprio
    )
    dataset.print_statistics()

    # Step 2: Generate training data
    print("\nStep 2: Generating training samples...")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")
    print(f"  Boundary tolerance: {boundary_tolerance}")

    train_dataset = SegmentBoundaryDataset(
        dataset=dataset,
        window_size=window_size,
        stride=stride,
        max_token_length=max_token_length,
        boundary_tolerance=boundary_tolerance,
        device=device
    )

    # Optional: Save dataset for faster loading next time
    dataset_cache_path = output_dir / 'cached_dataset.pkl'
    output_dir.mkdir(parents=True, exist_ok=True)
    # train_dataset.save_to_disk(dataset_cache_path)

    # Step 3: Create data loader
    print("\nStep 3: Creating data loaders...")
    train_loader, _ = create_dataloaders(
        train_dataset,
        batch_size=batch_size,
        num_workers=4
    )
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(train_loader)}")

    # Step 4: Create model
    print("\nStep 4: Creating model...")
    model = create_model(
        vocab_size=32000,  # Fast-DROID tokenizer
        max_token_length=max_token_length,
        hidden_dims=[512, 256, 128],
        dropout=0.3,
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")

    # Step 5: Setup training
    print("\nStep 5: Setting up training...")

    # Use Focal Loss to handle class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print("  Loss function: Focal Loss (alpha=0.25, gamma=2.0)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    print(f"  Optimizer: AdamW (lr={learning_rate})")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    print("  Scheduler: ReduceLROnPlateau")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # Add validation loader if you have val data
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir / 'checkpoints',
        log_dir=output_dir / 'logs'
    )

    # Step 6: Train
    print(f"\nStep 6: Training for {num_epochs} epochs...")
    print("="*70)

    trainer.train(num_epochs=num_epochs, save_every=5)

    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print(f"Logs saved to: {output_dir / 'logs'}")
    print(f"\nView training progress with: tensorboard --logdir {output_dir / 'logs'}")
    print("="*70)


if __name__ == "__main__":
    main()

"""
Train the GNN model for clash detection
"""
import sys
from pathlib import Path
import argparse
import pickle
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn_model import ClashDetectionModel


def train_gnn(training_data_file: str = 'data/training_data.pkl',
             output_model: str = 'models/saved_models/clash_gnn.pth',
             epochs: int = 100,
             learning_rate: float = 0.001,
             hidden_channels: int = 256,  # Increased from 128 for more capacity
             num_layers: int = 4,          # Increased from 3 for deeper network
             heads: int = 4,
             dropout: float = 0.3):        # Increased from 0.2 for better generalization
    """
    Train the GNN model on prepared training data.
    
    Args:
        training_data_file: Pickle file with training graphs
        output_model: Where to save the trained model
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        hidden_channels: Hidden dimension size
        num_layers: Number of GAT layers
        heads: Number of attention heads
        dropout: Dropout rate
    """
    print(f"\n{'='*60}")
    print(f"Training GNN Model")
    print(f"{'='*60}")
    
    # Load training data
    print(f"\nLoading training data from: {training_data_file}")
    with open(training_data_file, 'rb') as f:
        training_data = pickle.load(f)
    
    # Handle both old and new data format
    if isinstance(training_data, dict) and 'samples' in training_data:
        training_samples = training_data['samples']
        normalization = training_data.get('normalization', None)
        if normalization:
            print(f"  Loaded pre-normalized data with saved normalization parameters")
    else:
        # Old format - just a list of samples
        training_samples = training_data
        normalization = None
        print(f"  Loaded data in old format (no normalization)")
    
    print(f"Loaded {len(training_samples)} training samples")
    
    if len(training_samples) == 0:
        print("ERROR: No training samples found!")
        return
    
    # Initialize model
    sample_graph = training_samples[0]['graph']
    in_channels = sample_graph.num_node_features
    
    print(f"\nModel Configuration:")
    print(f"  Input channels: {in_channels}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Attention heads: {heads}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning rate: {learning_rate}")
    
    # Calculate class weights for balanced training
    print(f"\nCalculating class weights for balanced training...")
    total_clashes = sum(sample['labels'].sum().item() for sample in training_samples)
    total_edges = sum(sample['graph'].num_edges for sample in training_samples)
    total_non_clashes = total_edges - total_clashes
    
    # Weight inversely proportional to class frequency
    # pos_weight should be: (# non-clashes) / (# clashes) to balance the classes
    # Higher pos_weight = penalize false negatives more (missing clashes)
    clash_ratio = total_clashes / total_edges if total_edges > 0 else 0.5
    
    print(f"  Class distribution: {total_clashes}/{total_edges} clashes ({clash_ratio*100:.1f}%)")
    
    # AGGRESSIVE: Use 2.0x the balanced weight to heavily penalize missing clashes
    # In clash detection, false negatives (missing clashes) are worse than false positives
    base_weight = total_non_clashes / total_clashes if total_clashes > 0 else 1.0
    pos_weight_value = base_weight * 2.0  # Multiply by 2.0 for balanced recall/precision
    
    print(f"  Base pos_weight: {base_weight:.3f}")
    print(f"  Using AGGRESSIVE pos_weight={pos_weight_value:.3f} (2.0x - prioritizes finding clashes)")
    
    model = ClashDetectionModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        learning_rate=learning_rate,
        pos_weight=pos_weight_value
    )
    
    # Note: Feature normalization is now handled in prepare_training_data.py
    # Data is already normalized when loaded
    
    # Training loop with early stopping
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Early stopping patience: 50 epochs")
    print(f"{'='*60}\n")
    
    losses = []
    best_loss = float('inf')
    best_epoch = 0
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for sample in training_samples:
            graph = sample['graph']
            labels = sample['labels']
            
            # Train step
            loss = model.train_step(graph, labels)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(training_samples)
        losses.append(avg_loss)
        
        # Save best model and check early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model immediately
            output_path = Path(output_model)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(output_model)
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} (epoch {best_epoch}) | Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"  No improvement for {patience} epochs")
            break
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Final Loss: {avg_loss:.4f}")
    print(f"  Best Loss: {best_loss:.4f} (epoch {best_epoch})")
    print(f"  Total epochs: {epoch+1}")
    print(f"  Model saved to: {output_model}")
    
    # Save training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('GNN Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_path.parent / f'training_loss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_file, dpi=150)
    print(f"  Training curve saved to: {plot_file}")
    print(f"{'='*60}\n")
    
    return model, losses


def main():
    parser = argparse.ArgumentParser(description='Train GNN model for clash detection')
    parser.add_argument('--input', type=str, default='data/training_data.pkl',
                       help='Input training data file')
    parser.add_argument('--output', type=str, default='models/saved_models/clash_gnn.pth',
                       help='Output model file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden', type=int, default=128,
                       help='Hidden channels')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of GAT layers')
    parser.add_argument('--heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    args = parser.parse_args()
    
    train_gnn(
        training_data_file=args.input,
        output_model=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        heads=args.heads,
        dropout=args.dropout
    )


if __name__ == '__main__':
    main()

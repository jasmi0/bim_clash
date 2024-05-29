"""
Evaluate trained GNN model performance
"""
import sys
from pathlib import Path
import argparse
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn_model import ClashDetectionModel


def evaluate_gnn(model_file: str, test_data_file: str):
    """
    Evaluate GNN model performance on test data.
    
    Args:
        model_file: Path to trained model
        test_data_file: Path to test data pickle file
    """
    print(f"\n{'='*60}")
    print(f"Evaluating GNN Model")
    print(f"{'='*60}")
    
    # Load test data
    print(f"\nLoading test data from: {test_data_file}")
    with open(test_data_file, 'rb') as f:
        test_data = pickle.load(f)
    
    # Handle both old and new data format
    if isinstance(test_data, dict) and 'samples' in test_data:
        test_samples = test_data['samples']
        normalization = test_data.get('normalization', None)
    else:
        # Old format - just a list of samples
        test_samples = test_data
        normalization = None
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # Load model
    print(f"\nLoading model from: {model_file}")
    sample_graph = test_samples[0]['graph']
    
    model = ClashDetectionModel(
        in_channels=sample_graph.num_node_features,
        hidden_channels=128,
        num_layers=3
    )
    model.load_model(model_file)
    
    # Evaluate
    print(f"\nRunning evaluation...\n")
    
    all_predictions = []
    all_labels = []
    
    for i, sample in enumerate(test_samples):
        graph = sample['graph']
        labels = sample['labels']
        
        # Get predictions
        predictions = model.predict(graph)
        
        # Convert to binary (threshold at 0.5)
        # predictions is already numpy array from model.predict()
        binary_preds = (predictions > 0.5).astype(float)
        
        all_predictions.extend(binary_preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        
        print(f"Sample {i+1}/{len(test_samples)}: {graph.num_edges} edges")
    
    # Calculate metrics
    all_predictions = [int(p) for p in all_predictions]
    all_labels = [int(l) for l in all_labels]
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                No Clash  Clash")
    print(f"Actual No Clash {cm[0,0]:8d}  {cm[0,1]:5d}")
    print(f"       Clash    {cm[1,0]:8d}  {cm[1,1]:5d}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Clash', 'Clash'],
                yticklabels=['No Clash', 'Clash'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    plot_file = Path(model_file).parent / 'confusion_matrix.png'
    plt.savefig(plot_file, dpi=150)
    print(f"\nConfusion matrix saved to: {plot_file}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained GNN model')
    parser.add_argument('--model', type=str, default='models/saved_models/clash_gnn.pth',
                       help='Trained model file')
    parser.add_argument('--test-data', type=str, default='data/training_data.pkl',
                       help='Test data file')
    
    args = parser.parse_args()
    
    evaluate_gnn(args.model, args.test_data)


if __name__ == '__main__':
    main()

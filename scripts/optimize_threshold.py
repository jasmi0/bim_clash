"""
Optimize decision threshold for better F1 score
"""
import sys
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn_model import ClashDetectionModel

def find_optimal_threshold():
    """Find the optimal classification threshold"""
    
    # Load test data
    test_data_file = 'data/training_data.pkl'
    model_file = 'models/saved_models/clash_gnn.pth'
    
    with open(test_data_file, 'rb') as f:
        test_data = pickle.load(f)
    
    if isinstance(test_data, dict) and 'samples' in test_data:
        test_samples = test_data['samples']
    else:
        test_samples = test_data
    
    # Load model
    sample_graph = test_samples[0]['graph']
    model = ClashDetectionModel(
        in_channels=sample_graph.num_node_features,
        hidden_channels=256,
        num_layers=4,
        heads=4,
        dropout=0.3
    )
    model.load_model(model_file)
    
    # Collect predictions
    all_probs = []
    all_labels = []
    
    for sample in test_samples:
        graph = sample['graph']
        labels = sample['labels']
        
        predictions = model.predict(graph)
        all_probs.extend(predictions.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)
    
    # Try different thresholds
    print("\\n" + "="*70)
    print(" "*15 + "THRESHOLD OPTIMIZATION")
    print("="*70)
    print("\\nSearching for optimal classification threshold...\\n")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-"*70)
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in np.arange(0.3, 0.71, 0.02):
        preds = (all_probs > threshold).astype(int)
        
        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        marker = " ⭐" if f1 > best_f1 else ""
        print(f"{threshold:<12.2f} {acc*100:<11.2f}% {prec*100:<11.2f}% {rec*100:<11.2f}% {f1*100:<11.2f}%{marker}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'threshold': threshold
            }
    
    print("="*70)
    print(f"\\n🎯 OPTIMAL THRESHOLD: {best_threshold:.2f}\\n")
    print(f"  Accuracy:  {best_metrics['accuracy']*100:6.2f}%")
    print(f"  Precision: {best_metrics['precision']*100:6.2f}%")
    print(f"  Recall:    {best_metrics['recall']*100:6.2f}%")
    print(f"  F1 Score:  {best_metrics['f1']*100:6.2f}%")
    print("\\n" + "="*70)
    
    return best_threshold, best_metrics

if __name__ == '__main__':
    find_optimal_threshold()

"""
Display GNN model performance metrics
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.gnn_model import ClashDetectionModel

def show_model_metrics():
    """Display comprehensive model performance metrics"""
    
    print("\n" + "="*70)
    print(" "*20 + "GNN MODEL PERFORMANCE REPORT")
    print("="*70 + "\n")
    
    # Load test data
    test_data_file = 'data/training_data.pkl'
    model_file = 'models/saved_models/clash_gnn.pth'
    
    print(f"Model: {model_file}")
    print(f"Test Data: {test_data_file}\n")
    
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
    
    # Load model with improved architecture
    sample_graph = test_samples[0]['graph']
    model = ClashDetectionModel(
        in_channels=sample_graph.num_node_features,
        hidden_channels=256,  # Improved from 128
        num_layers=4,          # Improved from 3
        heads=4,
        dropout=0.3            # Improved from 0.2
    )
    model.load_model(model_file)
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    all_probs = []
    
    for sample in test_samples:
        graph = sample['graph']
        labels = sample['labels']
        
        predictions = model.predict(graph)
        binary_preds = (predictions > 0.5).astype(float)
        
        all_predictions.extend(binary_preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        all_probs.extend(predictions.flatten())
    
    all_predictions = [int(p) for p in all_predictions]
    all_labels = [int(l) for l in all_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Display metrics
    print("OVERALL METRICS")
    print("-" * 70)
    print(f"  Accuracy:   {accuracy*100:6.2f}%  (correctly classified edges)")
    print(f"  Precision:  {precision*100:6.2f}%  (of predicted clashes, how many are real)")
    print(f"  Recall:     {recall*100:6.2f}%  (of real clashes, how many were found)")
    print(f"  F1 Score:   {f1*100:6.2f}%  (harmonic mean of precision & recall)")
    print()
    
    # Confusion matrix
    print("CONFUSION MATRIX")
    print("-" * 70)
    print(f"                      Predicted")
    print(f"                 No Clash    Clash     Total")
    print(f"  Actual")
    print(f"    No Clash      {cm[0,0]:5d}      {cm[0,1]:5d}      {cm[0,0]+cm[0,1]:5d}")
    print(f"    Clash         {cm[1,0]:5d}      {cm[1,1]:5d}      {cm[1,0]+cm[1,1]:5d}")
    print(f"    Total         {cm[0,0]+cm[1,0]:5d}      {cm[0,1]+cm[1,1]:5d}      {len(all_labels):5d}")
    print()
    
    # True/False Positives/Negatives
    tn, fp, fn, tp = cm.ravel()
    print("DETAILED BREAKDOWN")
    print("-" * 70)
    print(f"  True Negatives (TN):   {tn:4d}  (correctly predicted no clash)")
    print(f"  True Positives (TP):   {tp:4d}  (correctly predicted clash)")
    print(f"  False Negatives (FN):  {fn:4d}  (missed clashes - bad!)")
    print(f"  False Positives (FP):  {fp:4d}  (false alarms)")
    print()
    
    # Prediction distribution
    import numpy as np
    probs_array = np.array(all_probs)
    
    print("PREDICTION CONFIDENCE DISTRIBUTION")
    print("-" * 70)
    print(f"  Very High (>90%):  {(probs_array > 0.9).sum():4d} edges")
    print(f"  High (70-90%):     {((probs_array > 0.7) & (probs_array <= 0.9)).sum():4d} edges")
    print(f"  Medium (50-70%):   {((probs_array > 0.5) & (probs_array <= 0.7)).sum():4d} edges")
    print(f"  Low (30-50%):      {((probs_array > 0.3) & (probs_array <= 0.5)).sum():4d} edges")
    print(f"  Very Low (<30%):   {(probs_array <= 0.3).sum():4d} edges")
    print()
    
    # Dataset info
    print("DATASET INFORMATION")
    print("-" * 70)
    print(f"  Total samples:     {len(test_samples)}")
    print(f"  Total edges:       {len(all_labels)}")
    print(f"  Actual clashes:    {sum(all_labels)} ({sum(all_labels)/len(all_labels)*100:.1f}%)")
    print(f"  Non-clashes:       {len(all_labels)-sum(all_labels)} ({(len(all_labels)-sum(all_labels))/len(all_labels)*100:.1f}%)")
    print()
    
    # Model info
    total_params = sum(p.numel() for p in model.model.parameters())
    print("MODEL INFORMATION")
    print("-" * 70)
    print(f"  Architecture:      Graph Attention Network (GAT)")
    print(f"  Layers:            3")
    print(f"  Hidden channels:   128")
    print(f"  Attention heads:   4")
    print(f"  Total parameters:  {total_params:,}")
    print()
    
    # Interpretation
    print("INTERPRETATION")
    print("-" * 70)
    if accuracy > 0.9:
        print("  ✓ Excellent accuracy - model is performing very well")
    elif accuracy > 0.8:
        print("  ✓ Good accuracy - model is reliable")
    elif accuracy > 0.7:
        print("  ~ Fair accuracy - model is decent but could improve")
    else:
        print("  ✗ Low accuracy - model needs more training data or tuning")
    
    if precision > 0.9:
        print("  ✓ High precision - few false alarms")
    elif precision > 0.7:
        print("  ~ Moderate precision - some false alarms")
    else:
        print("  ✗ Low precision - many false alarms")
    
    if recall > 0.9:
        print("  ✓ High recall - finding most real clashes")
    elif recall > 0.7:
        print("  ~ Moderate recall - missing some clashes")
    else:
        print("  ✗ Low recall - missing many clashes (need more training)")
    
    if f1 > 0.8:
        print("  ✓ Strong F1 score - good balance")
    elif f1 > 0.6:
        print("  ~ Moderate F1 score - room for improvement")
    else:
        print("  ✗ Low F1 score - imbalanced performance")
    
    print()
    
    # Recommendations
    print("RECOMMENDATIONS")
    print("-" * 70)
    if recall < 0.5:
        print("  • Low recall - Add more training data with clash examples")
        print("  • Try increasing training epochs (--epochs 200)")
    if precision < 0.7:
        print("  • Low precision - Model needs better negative examples")
        print("  • Balance your training dataset better")
    if len(test_samples) < 10:
        print("  • Small dataset - Train on at least 10-20 IFC files")
        print("  • More data will significantly improve performance")
    if accuracy > 0.85 and f1 > 0.7:
        print("  ✓ Model is performing well!")
        print("  • Can use confidently in production")
        print("  • Continue training with more data to improve further")
    
    print()
    print("="*70)
    print()
    
    # Save report
    report_file = "models/saved_models/performance_report.txt"
    with open(report_file, 'w') as f:
        f.write("GNN MODEL PERFORMANCE REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy:  {accuracy*100:.2f}%\n")
        f.write(f"Precision: {precision*100:.2f}%\n")
        f.write(f"Recall:    {recall*100:.2f}%\n")
        f.write(f"F1 Score:  {f1*100:.2f}%\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN={tn}, FP={fp}\n")
        f.write(f"  FN={fn}, TP={tp}\n")
    
    print(f"Report saved to: {report_file}")

if __name__ == '__main__':
    show_model_metrics()

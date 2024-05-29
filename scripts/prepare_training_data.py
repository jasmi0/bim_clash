"""
Prepare training data from labeled clashes for GNN training
"""
import sys
from pathlib import Path
import argparse
import pickle
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_builder import BIMGraphBuilder
from src.clash_detector import ClashDetector


def prepare_training_data(labeled_data_file: str = 'data/labeled_data.pkl', 
                         output_file: str = 'data/training_data.pkl'):
    """
    Convert labeled clash data to PyTorch Geometric graphs for training.
    
    NEW APPROACH: Build graph first, then label edges directly using geometric detection.
    This ensures perfect alignment between graph edges and labels.
    
    Args:
        labeled_data_file: Pickle file with labeled data from auto_label_clashes.py
        output_file: Where to save prepared training graphs
    """
    print(f"\n{'='*60}")
    print(f"Preparing Training Data (Graph-First Labeling)")
    print(f"{'='*60}")
    
    # Load labeled data
    print(f"\nLoading labeled data from: {labeled_data_file}")
    with open(labeled_data_file, 'rb') as f:
        labeled_data = pickle.load(f)
    
    print(f"Loaded {len(labeled_data)} labeled datasets")
    
    # Convert to graphs with edge-based labeling
    training_samples = []
    
    for i, data in enumerate(labeled_data):
        print(f"\nProcessing dataset {i+1}/{len(labeled_data)}: {data['file']}")
        
        elements = data['elements']
        
        # Skip empty datasets
        if len(elements) == 0:
            print(f"  Skipping: No elements found")
            continue
        
        # STEP 1: Build graph first
        graph_builder = BIMGraphBuilder(elements)
        graph = graph_builder.build_graph()
        
        print(f"  Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        
        if graph.num_edges == 0:
            print(f"  Skipping: No edges in graph")
            continue
        
        # STEP 2: Run geometric clash detection
        config = {
            'clash_detection': {
                'distance_threshold': 0.01,  # 1cm clash tolerance
                'severity_levels': {
                    'critical': 0.0,
                    'high': 0.02,
                    'medium': 0.05,
                    'low': 0.10
                }
            }
        }
        
        detector = ClashDetector(elements, config)
        clashes = detector.detect_clashes()
        
        # Create a set of clashing pairs for fast lookup
        clashing_pairs = set()
        for clash in clashes:
            pair1 = (clash.element1_guid, clash.element2_guid)
            pair2 = (clash.element2_guid, clash.element1_guid)
            clashing_pairs.add(pair1)
            clashing_pairs.add(pair2)
        
        print(f"  Geometric clashes found: {len(clashes)}")
        
        # STEP 3: Label each graph edge based on geometric detection
        num_edges = graph.edge_index.shape[1]
        labels = torch.zeros(num_edges, dtype=torch.float32)
        
        clash_count = 0
        for edge_idx in range(num_edges):
            src_idx = graph.edge_index[0, edge_idx].item()
            dst_idx = graph.edge_index[1, edge_idx].item()
            
            src_guid = elements[src_idx]['guid']
            dst_guid = elements[dst_idx]['guid']
            
            # Check if this edge represents a clash
            pair = (src_guid, dst_guid)
            if pair in clashing_pairs:
                labels[edge_idx] = 1.0
                clash_count += 1
        
        non_clash_count = num_edges - clash_count
        clash_ratio = clash_count / num_edges * 100 if num_edges > 0 else 0
        
        print(f"  Edge labels: {clash_count} clashes, {non_clash_count} non-clashes ({clash_ratio:.1f}% clashes)")
        
        training_samples.append({
            'graph': graph,
            'labels': labels,
            'file': data['file']
        })
    
    # Normalize node features to prevent gradient collapse
    print(f"\n{'='*60}")
    print(f"Normalizing node features...")
    print(f"{'='*60}")
    
    all_features = []
    for sample in training_samples:
        all_features.append(sample['graph'].x)
    
    all_features_tensor = torch.cat(all_features, dim=0)
    feature_mean = all_features_tensor.mean(dim=0, keepdim=True)
    feature_std = all_features_tensor.std(dim=0, keepdim=True) + 1e-6  # Prevent division by zero
    
    print(f"  Feature stats before normalization:")
    print(f"    Min: {all_features_tensor.min().item():.2f}")
    print(f"    Max: {all_features_tensor.max().item():.2f}")
    print(f"    Mean range: [{feature_mean.min().item():.2f}, {feature_mean.max().item():.2f}]")
    print(f"    Std range: [{feature_std.min().item():.2f}, {feature_std.max().item():.2f}]")
    
    # Apply normalization
    for sample in training_samples:
        sample['graph'].x = (sample['graph'].x - feature_mean) / feature_std
    
    normalized_features = torch.cat([sample['graph'].x for sample in training_samples], dim=0)
    print(f"  Feature stats after normalization:")
    print(f"    Min: {normalized_features.min().item():.2f}")
    print(f"    Max: {normalized_features.max().item():.2f}")
    print(f"    Mean: ~0.00, Std: ~1.00")
    
    # Save training data with normalization parameters
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    training_data = {
        'samples': training_samples,
        'normalization': {
            'mean': feature_mean,
            'std': feature_std
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(training_data, f)
    
    print(f"\n{'='*60}")
    print(f"Training data saved to: {output_file}")
    print(f"Total training samples: {len(training_samples)}")
    total_edges = sum(s['graph'].num_edges for s in training_samples)
    total_clashes = sum(s['labels'].sum().item() for s in training_samples)
    print(f"Total edges: {total_edges}")
    print(f"Total clashes: {int(total_clashes)}")
    print(f"Clash ratio: {total_clashes/total_edges*100:.2f}%")
    print(f"Normalization parameters saved for inference")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for GNN')
    parser.add_argument('--input', type=str, default='data/labeled_data.pkl', 
                       help='Input labeled data file')
    parser.add_argument('--output', type=str, default='data/training_data.pkl', 
                       help='Output training data file')
    
    args = parser.parse_args()
    
    prepare_training_data(args.input, args.output)


if __name__ == '__main__':
    main()

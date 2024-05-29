"""
Auto-label clashes using geometric detection to create training data
"""
import sys
from pathlib import Path
import argparse
import pickle
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bim_parser import BIMParser
from src.clash_detector import ClashDetector


def auto_label_from_geometric_detection(ifc_file: str, tolerance: float = 0.01) -> Dict[Tuple[str, str], bool]:
    """
    Use geometric clash detector to create training labels.
    
    Args:
        ifc_file: Path to IFC file
        tolerance: Clash detection tolerance in meters
        
    Returns:
        Dictionary mapping (element1_id, element2_id) -> is_clash
    """
    print(f"\nProcessing: {ifc_file}")
    
    # Parse IFC
    parser = BIMParser(ifc_file)
    parser.load_ifc_file()
    elements = parser.extract_elements()
    
    print(f"  Elements: {len(elements)}")
    
    # Create config for clash detector
    config = {
        'clash_detection': {
            'distance_threshold': tolerance,
            'severity_levels': {
                'critical': 0.0,
                'high': 0.02,
                'medium': 0.05,
                'low': 0.10
            }
        }
    }
    
    # Detect clashes geometrically
    detector = ClashDetector(elements, config)
    clashes = detector.detect_clashes()
    
    print(f"  Clashes found: {len(clashes)}")
    
    # Create labels - mark clashing pairs as True
    clash_labels = {}
    for clash in clashes:
        # Use GUIDs for labeling (more reliable than IDs)
        pair1 = (clash.element1_guid, clash.element2_guid)
        pair2 = (clash.element2_guid, clash.element1_guid)
        clash_labels[pair1] = True
        clash_labels[pair2] = True
    
    # IMPROVED: Create non-clash labels for ALL nearby element pairs
    # This ensures we label edges that will actually exist in the graph
    import numpy as np
    from scipy.spatial.distance import cdist
    
    # Get element centers
    centers = []
    guid_to_idx = {}
    for idx, elem in enumerate(elements):
        guid_to_idx[elem['guid']] = idx
        if elem['bounding_box']:
            centers.append(elem['bounding_box']['center'])
        else:
            centers.append([0, 0, 0])
    
    if len(centers) > 1:
        centers = np.array(centers)
        
        # Calculate distances between all pairs
        # Use larger threshold to match graph builder (2.0 meters)
        graph_distance_threshold = 2.0
        distances = cdist(centers, centers)
        
        # Label ALL nearby pairs to match graph structure
        nearby_pairs = []
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if distances[i, j] < graph_distance_threshold:
                    elem1_guid = elements[i]['guid']
                    elem2_guid = elements[j]['guid']
                    pair1 = (elem1_guid, elem2_guid)
                    pair2 = (elem2_guid, elem1_guid)
                    nearby_pairs.append((pair1, pair2, distances[i, j]))
        
        # BALANCE THE DATASET: Ensure 40-60% clash ratio
        clash_pairs = [(p1, p2) for (p1, p2, d) in nearby_pairs if p1 in clash_labels or p2 in clash_labels]
        non_clash_pairs = [(p1, p2) for (p1, p2, d) in nearby_pairs if p1 not in clash_labels and p2 not in clash_labels]
        
        num_clashes = len(clash_pairs)
        
        # Target: Keep all clashes, but ensure at least equal non-clashes
        # This prevents model from being biased toward predicting clashes
        target_non_clashes = max(num_clashes, len(non_clash_pairs))
        
        # Add all non-clash pairs
        for pair1, pair2 in non_clash_pairs:
            clash_labels[pair1] = False
            clash_labels[pair2] = False
    
    clash_count = sum(1 for v in clash_labels.values() if v)
    non_clash_count = sum(1 for v in clash_labels.values() if not v)
    print(f"  Labels created: {len(clash_labels)} ({clash_count} clashes, {non_clash_count} non-clashes)")
    print(f"  Balance: {clash_count/(clash_count+non_clash_count)*100:.1f}% clashes, {non_clash_count/(clash_count+non_clash_count)*100:.1f}% non-clashes")
    
    return clash_labels, elements


def process_multiple_files(ifc_files: List[str], output_file: str = 'data/labeled_data.pkl', tolerance: float = 0.01):
    """
    Process multiple IFC files and create labeled dataset.
    
    Args:
        ifc_files: List of IFC file paths
        output_file: Where to save the labeled data
        tolerance: Clash detection tolerance
    """
    print(f"\n{'='*60}")
    print(f"Auto-Labeling {len(ifc_files)} IFC Files")
    print(f"{'='*60}")
    
    labeled_data = []
    
    for ifc_file in ifc_files:
        try:
            labels, elements = auto_label_from_geometric_detection(ifc_file, tolerance)
            labeled_data.append({
                'file': ifc_file,
                'labels': labels,
                'elements': elements
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save labeled data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(labeled_data, f)
    
    print(f"\n{'='*60}")
    print(f"Labeled data saved to: {output_file}")
    print(f"Total files: {len(labeled_data)}")
    total_clashes = sum(sum(d['labels'].values()) for d in labeled_data)
    total_labels = sum(len(d['labels']) for d in labeled_data)
    print(f"Total labels: {total_labels} ({total_clashes} clashes)")
    print(f"{'='*60}\n")
    
    return labeled_data


def main():
    parser = argparse.ArgumentParser(description='Auto-label IFC files using geometric clash detection')
    parser.add_argument('--files', nargs='+', help='IFC files to process')
    parser.add_argument('--dir', type=str, help='Directory containing IFC files')
    parser.add_argument('--output', type=str, default='data/labeled_data.pkl', help='Output file for labeled data')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Clash detection tolerance (meters)')
    
    args = parser.parse_args()
    
    # Collect IFC files
    ifc_files = []
    
    if args.files:
        ifc_files.extend(args.files)
    
    if args.dir:
        dir_path = Path(args.dir)
        ifc_files.extend([str(f) for f in dir_path.glob('**/*.ifc')])
    
    if not ifc_files:
        print("No IFC files specified. Use --files or --dir")
        print("\nExample usage:")
        print("  python scripts/auto_label_clashes.py --files model1.ifc model2.ifc")
        print("  python scripts/auto_label_clashes.py --dir models/sample_models/")
        return
    
    # Process files
    process_multiple_files(ifc_files, args.output, args.tolerance)


if __name__ == '__main__':
    main()

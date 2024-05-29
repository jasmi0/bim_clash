"""
Run the complete GNN training pipeline: label -> prepare -> train
"""
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.auto_label_clashes import process_multiple_files
from scripts.prepare_training_data import prepare_training_data
from scripts.train_gnn import train_gnn


def run_full_pipeline(ifc_dir: str = None,
                     ifc_files: list = None,
                     tolerance: float = 0.01,
                     epochs: int = 100,
                     learning_rate: float = 0.001):
    """
    Run the complete training pipeline.
    
    Args:
        ifc_dir: Directory containing IFC files
        ifc_files: List of specific IFC files
        tolerance: Clash detection tolerance
        epochs: Training epochs
        learning_rate: Learning rate
    """
    print(f"\n{'='*60}")
    print(f"FULL GNN TRAINING PIPELINE")
    print(f"{'='*60}\n")
    
    # Collect IFC files
    files = []
    if ifc_files:
        files.extend(ifc_files)
    if ifc_dir:
        dir_path = Path(ifc_dir)
        files.extend([str(f) for f in dir_path.glob('**/*.ifc')])
    
    if not files:
        print("ERROR: No IFC files provided!")
        print("Use --dir or --files")
        return
    
    print(f"Found {len(files)} IFC files\n")
    
    # Step 1: Auto-label clashes
    print("STEP 1: Auto-labeling clashes using geometric detection")
    print("-" * 60)
    labeled_data = process_multiple_files(files, 'data/labeled_data.pkl', tolerance)
    
    if not labeled_data:
        print("ERROR: No data was labeled!")
        return
    
    # Step 2: Prepare training data
    print("\nSTEP 2: Preparing training data (building graphs)")
    print("-" * 60)
    prepare_training_data('data/labeled_data.pkl', 'data/training_data.pkl')
    
    # Step 3: Train GNN
    print("\nSTEP 3: Training GNN model")
    print("-" * 60)
    model, losses = train_gnn(
        training_data_file='data/training_data.pkl',
        output_model='models/saved_models/clash_gnn.pth',
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"\nYour trained GNN model is ready:")
    print(f"  models/saved_models/clash_gnn.pth")
    print(f"\nYou can now use it in the Streamlit app!")
    print(f"  streamlit run app.py")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Run full GNN training pipeline')
    parser.add_argument('--dir', type=str, help='Directory containing IFC files')
    parser.add_argument('--files', nargs='+', help='Specific IFC files')
    parser.add_argument('--tolerance', type=float, default=0.01, 
                       help='Clash detection tolerance (meters)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    if not args.dir and not args.files:
        print("\nERROR: Must specify --dir or --files")
        print("\nExamples:")
        print("  python scripts/run_full_pipeline.py --dir models/sample_models/")
        print("  python scripts/run_full_pipeline.py --files model1.ifc model2.ifc --epochs 50")
        return
    
    run_full_pipeline(
        ifc_dir=args.dir,
        ifc_files=args.files,
        tolerance=args.tolerance,
        epochs=args.epochs,
        learning_rate=args.lr
    )


if __name__ == '__main__':
    main()

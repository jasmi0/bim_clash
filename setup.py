"""
Setup script for BIM Clash Detection project
"""
import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")


def check_python_version():
    """Check if Python version is 3.8+"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8+ required")
        print("Please upgrade Python and try again")
        return False
    
    print("Python version is compatible")
    return True


def install_dependencies():
    """Install required dependencies"""
    print_header("Installing Dependencies")
    
    print("This will install all required packages from requirements.txt")
    
    try:
        # Upgrade pip first
        print("\n Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("\n Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("\n All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n Error installing dependencies: {str(e)}")
        print("\n Troubleshooting:")
        print("1. Make sure you've activated the virtual environment")
        print("2. Try running: pip install --upgrade pip setuptools wheel")
        print("3. Check INSTALLATION.md for platform-specific instructions")
        return False


def create_directories():
    """Create required directories"""
    print_header("Creating Required Directories")
    
    directories = [
        "data/uploaded_files",
        "data/processed",
        "models/saved_models",
        "logs"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f" Created: {dir_path}")
        else:
            print(f"✓ Exists: {dir_path}")
    
    return True


def verify_installation():
    """Verify installation by running test script"""
    print_header("Verifying Installation")
    
    print("Running installation tests...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "test_installation.py"],
            capture_output=False,
            text=True
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f" Error running tests: {str(e)}")
        return False


def main():
    """Main setup function"""
    print("\n" + "🏗️ " * 20)
    print("  BIM CLASH DETECTION - SETUP SCRIPT")
    print("🏗️ " * 20)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create directories
    if not create_directories():
        print("\n Warning: Could not create all directories")
    
    # Step 3: Ask user about installation
    print_header("Dependency Installation")
    print("This will install all required packages.")
    print("\nPackages to be installed:")
    print("- streamlit, pandas, numpy")
    print("- ifcopenshell (BIM processing)")
    print("- torch, torch-geometric (AI/ML)")
    print("- plotly (visualization)")
    print("- and more...")
    
    response = input("\n📦 Install dependencies now? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        if not install_dependencies():
            print("\n Setup failed during dependency installation")
            sys.exit(1)
    else:
        print("\n  Skipping dependency installation")
        print("Run 'pip install -r requirements.txt' manually later")
    
    # Step 4: Verify installation
    print_header("Final Steps")
    response = input(" Run installation tests? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        if verify_installation():
            print("\n Setup completed successfully!")
        else:
            print("\n  Setup completed with warnings")
            print("Please review the test results above")
    
    # Print next steps
    print_header("Next Steps")
    print("1. Review README.md for usage instructions")
    print("2. Run the application:")
    print("   streamlit run app.py")
    print("\n3. Access the app at http://localhost:8501")
    print("\n4. Upload an IFC file to begin clash detection")
    print("\n5. Check example_usage.py for API usage examples")
    
    print("\n Documentation:")
    print("   - README.md - Full documentation")
    print("   - INSTALLATION.md - Installation guide")
    print("   - config.yaml - Configuration options")
    
    print("\n" + "🏗️ " * 20)
    print("  Happy Clash Detecting!")
    print("🏗️ " * 20 + "\n")


if __name__ == "__main__":
    main()

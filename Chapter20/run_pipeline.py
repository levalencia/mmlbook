#!/usr/bin/env python3
"""
Simple wrapper to run the complete model pipeline
"""

import sys
import subprocess
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_pipeline():
    """Run the full model pipeline"""
    print("Starting model pipeline...")
    try:
        subprocess.run([sys.executable, "fullscript.py"], check=True)
        print("✓ Pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        return False

def main():
    print("🚀 PyTorch Model Development Pipeline")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("fullscript.py"):
        print("❌ fullscript.py not found. Please run this from the Chapter20 directory.")
        sys.exit(1)
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found. Please run this from the Chapter20 directory.")
        sys.exit(1)
    
    # Ask user if they want to install requirements
    install_deps = input("Install/update requirements? (y/n): ").lower().strip()
    if install_deps in ['y', 'yes']:
        if not install_requirements():
            sys.exit(1)
    
    print("\nStarting pipeline...")
    print("This will:")
    print("  • Train MNIST models (SGD and Adam)")
    print("  • Optimize models (quantization, pruning)")
    print("  • Convert to TorchScript and ONNX")
    print("  • Version models")
    print("  • Attempt Azure ML deployment (if configured)")
    print()
    
    proceed = input("Continue? (y/n): ").lower().strip()
    if proceed not in ['y', 'yes']:
        print("Pipeline cancelled.")
        sys.exit(0)
    
    success = run_pipeline()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Pipeline completed successfully!")
        print("\nCheck the following outputs:")
        print("  • model_pipeline.log - Full execution log")
        print("  • models/ - Saved model files")
        print("  • model_versions/ - Versioned models")
    else:
        print("❌ Pipeline failed. Check model_pipeline.log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
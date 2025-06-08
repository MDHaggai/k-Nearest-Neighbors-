"""
Quick demo script to test the automatic results saving functionality
"""

import os
import sys
sys.path.append('src')

from src.result_saver import ResultSaver
import matplotlib.pyplot as plt
import numpy as np

def demo_result_saver():
    """Demonstrate the result saver functionality"""
    print("Testing Result Saver Functionality")
    print("=" * 40)
    
    # Initialize result saver
    saver = ResultSaver(base_dir='results')
    
    # Create a demo plot
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.title('Demo Plot - Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    saver.save_plot('demo_plot', 'Demo Plot - Sine Wave')
    plt.show()
    
    # Save some demo metrics
    demo_metrics = {
        'accuracy': 0.95,
        'precision': 0.93,
        'recall': 0.97,
        'f1_score': 0.95,
        'confusion_matrix': [[85, 5], [3, 87]]
    }
    
    saver.save_metrics(demo_metrics, 'demo_metrics', 'demo')
    
    # Create summary report
    summary = saver.create_summary_report()
    
    print(f"\nDemo completed successfully!")
    print(f"Results saved to: {saver.base_dir}")
    print(f"Plots: {saver.plots_dir}")
    print(f"Metrics: {saver.metrics_dir}")
    
    return saver

def check_results_structure():
    """Check the results folder structure"""
    print("Checking Results Folder Structure")
    print("=" * 40)
    
    base_dir = 'results'
    if os.path.exists(base_dir):
        print(f"✅ Results directory exists: {base_dir}")
        
        # Check subdirectories
        for subdir in ['plots', 'metrics']:
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"✅ {subdir} directory exists: {subdir_path}")
                
                # List files in subdirectory
                files = os.listdir(subdir_path)
                if files:
                    print(f"   Files in {subdir}: {len(files)}")
                    for file in files[:5]:  # Show first 5 files
                        print(f"     - {file}")
                    if len(files) > 5:
                        print(f"     ... and {len(files) - 5} more files")
                else:
                    print(f"   No files in {subdir} yet")
            else:
                print(f"❌ {subdir} directory missing: {subdir_path}")
    else:
        print(f"❌ Results directory missing: {base_dir}")
        print("Run the notebook first to create the results structure")

if __name__ == "__main__":
    print("k-NN Project Results Demo")
    print("=" * 50)
    
    # Check current structure
    check_results_structure()
    
    print("\n")
    
    # Run demo (optional - uncomment to test)
    # demo_result_saver()
    
    print("\nTo see the full results:")
    print("1. Run the Jupyter notebook: notebooks/knn_analysis.ipynb")
    print("2. All plots and metrics will be automatically saved")
    print("3. Check the results/ folder for organized output")

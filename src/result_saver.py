"""
Result Saver Utility for k-NN Analysis
Automatically saves plots, metrics, and analysis results to organized folder structure.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

class ResultSaver:
    """Utility class for automatically saving analysis results."""
    
    def __init__(self, base_dir: str = "../results"):
        """
        Initialize the ResultSaver with base directory for saving results.
        
        Args:
            base_dir: Base directory path for saving results
        """
        self.base_dir = base_dir
        self.plots_dir = os.path.join(base_dir, "plots")
        self.metrics_dir = os.path.join(base_dir, "metrics")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories if they don't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_summary = {}
        self.plot_counter = 0
        
    def save_plot(self, filename: str = None, title: str = None, 
                  format: str = 'png', dpi: int = 300, bbox_inches: str = 'tight'):
        """
        Save the current matplotlib figure to the plots directory.
        
        Args:
            filename: Custom filename for the plot (without extension)
            title: Title to use for filename generation if filename not provided
            format: Image format (png, jpg, svg, pdf)
            dpi: Resolution for saving
            bbox_inches: Bounding box setting
        """
        self.plot_counter += 1
        
        if filename is None:
            if title:
                # Clean title for filename
                clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_title = clean_title.replace(' ', '_')
                filename = f"{self.plot_counter:02d}_{clean_title}"
            else:
                filename = f"plot_{self.plot_counter:02d}"
        
        filepath = os.path.join(self.plots_dir, f"{filename}.{format}")
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format=format)
        print(f"Plot saved: {filepath}")
        
    def save_metrics(self, metrics: Dict[str, Any], filename: str = None, 
                     category: str = "general"):
        """
        Save metrics to JSON and CSV formats.
        
        Args:
            metrics: Dictionary of metrics to save
            filename: Custom filename for metrics
            category: Category of metrics (e.g., 'knn_evaluation', 'comparison')
        """
        if filename is None:
            filename = f"{category}_metrics_{self.timestamp}"
        
        # Save to JSON
        json_path = os.path.join(self.metrics_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=self._json_serializer)
        
        # Convert to DataFrame and save as CSV if possible
        try:
            if isinstance(metrics, dict):
                # Try to flatten nested dictionaries for CSV
                flattened = self._flatten_dict(metrics)
                df = pd.DataFrame([flattened])
            else:
                df = pd.DataFrame(metrics)
            
            csv_path = os.path.join(self.metrics_dir, f"{filename}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Metrics saved: {json_path} and {csv_path}")
        except Exception as e:
            print(f"Metrics saved to JSON: {json_path} (CSV conversion failed: {e})")
        
        # Add to summary
        self.metrics_summary[category] = metrics
        
    def save_dataset_info(self, datasets_info: Dict[str, Any]):
        """Save dataset information and statistics."""
        self.save_metrics(datasets_info, "dataset_information", "datasets")
        
    def save_knn_results(self, results: Dict[str, Any]):
        """Save k-NN evaluation results."""
        self.save_metrics(results, "knn_evaluation_results", "knn_evaluation")
        
    def save_comparison_results(self, results: Dict[str, Any]):
        """Save model comparison results."""
        self.save_metrics(results, "model_comparison_results", "model_comparison")
        
    def save_hyperparameter_results(self, results: Dict[str, Any]):
        """Save hyperparameter tuning results."""
        self.save_metrics(results, "hyperparameter_tuning", "hyperparameters")
        
    def save_distance_metrics_results(self, results: Dict[str, Any]):
        """Save distance metrics comparison results."""
        self.save_metrics(results, "distance_metrics_comparison", "distance_metrics")
        
    def save_dimensionality_analysis(self, results: Dict[str, Any]):
        """Save curse of dimensionality analysis results."""
        self.save_metrics(results, "dimensionality_analysis", "dimensionality")
        
    def create_summary_report(self):
        """Create a comprehensive summary report of all saved results."""
        summary = {
            "analysis_timestamp": self.timestamp,
            "total_plots_saved": self.plot_counter,
            "metrics_categories": list(self.metrics_summary.keys()),
            "results_summary": self.metrics_summary,
            "files_created": {
                "plots_directory": self.plots_dir,
                "metrics_directory": self.metrics_dir
            }
        }
        
        # Save summary report
        summary_path = os.path.join(self.base_dir, f"analysis_summary_{self.timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)
        
        print(f"\nAnalysis Summary Report created: {summary_path}")
        print(f"Total plots saved: {self.plot_counter}")
        print(f"Metrics categories: {len(self.metrics_summary)}")
        
        return summary
        
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float, str)):
                # Convert simple lists to comma-separated strings
                items.append((new_key, ','.join(map(str, v))))
            else:
                items.append((new_key, v))
        return dict(items)

# Convenience function for one-time plot saving
def save_current_plot(filename: str = None, title: str = None, results_dir: str = "../results"):
    """
    Convenience function to save the current plot without creating a ResultSaver instance.
    
    Args:
        filename: Custom filename for the plot
        title: Title to use for filename generation
        results_dir: Directory to save results
    """
    saver = ResultSaver(results_dir)
    saver.save_plot(filename, title)
    
# Enhanced plotting functions with automatic saving
def plot_and_save(plot_func, *args, filename: str = None, title: str = None, 
                  saver: ResultSaver = None, **kwargs):
    """
    Execute a plotting function and automatically save the result.
    
    Args:
        plot_func: Function that creates the plot
        *args: Arguments for the plotting function
        filename: Custom filename for saving
        title: Title for the plot
        saver: ResultSaver instance (creates new one if None)
        **kwargs: Keyword arguments for the plotting function
    """
    if saver is None:
        saver = ResultSaver()
    
    # Execute the plotting function
    result = plot_func(*args, **kwargs)
    
    # Set title if provided
    if title:
        plt.title(title)
    
    # Save the plot
    saver.save_plot(filename, title)
    
    # Show the plot
    plt.show()
    
    return result, saver

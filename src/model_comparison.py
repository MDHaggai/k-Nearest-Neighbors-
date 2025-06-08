"""
Comprehensive Model Comparison and Analysis
This module compares k-NN with other machine learning algorithms
and provides detailed analysis of performance across different datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import time
import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """
    Comprehensive model comparison class for evaluating k-NN against other algorithms
    """
    
    def __init__(self):
        self.models = {
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        self.results = {}
    
    def evaluate_models(self, X_train, X_test, y_train, y_test, dataset_name="Dataset"):
        """
        Evaluate all models on given dataset
        
        Parameters:
        -----------
        X_train, X_test : arrays
            Training and testing features
        y_train, y_test : arrays
            Training and testing targets
        dataset_name : str
            Name of the dataset for reporting
        """
        print(f"\nEvaluating models on {dataset_name}:")
        print("=" * 50)
        
        dataset_results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Time the training
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Time the prediction
            start_time = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            dataset_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'train_time': train_time,
                'predict_time': predict_time,
                'predictions': y_pred
            }
            
            print(f"  {name}: Accuracy = {accuracy:.4f}, CV = {cv_mean:.4f}±{cv_std:.4f}")
        
        self.results[dataset_name] = dataset_results
        return dataset_results
    
    def plot_model_comparison(self, dataset_name, metric='accuracy'):
        """Plot comparison of models for a specific dataset and metric"""
        if dataset_name not in self.results:
            print(f"No results found for {dataset_name}")
            return
        
        data = self.results[dataset_name]
        models = list(data.keys())
        values = [data[model][metric] for model in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        plt.title(f'{metric.capitalize()} Comparison - {dataset_name}')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_matrix(self):
        """Plot heatmap of model performance across all datasets"""
        if not self.results:
            print("No results to plot")
            return
        
        # Create performance matrix
        datasets = list(self.results.keys())
        models = list(self.models.keys())
        
        accuracy_matrix = np.zeros((len(datasets), len(models)))
        
        for i, dataset in enumerate(datasets):
            for j, model in enumerate(models):
                accuracy_matrix[i, j] = self.results[dataset][model]['accuracy']
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=models, yticklabels=datasets)
        plt.title('Model Performance Across Datasets (Accuracy)')
        plt.xlabel('Models')
        plt.ylabel('Datasets')
        plt.tight_layout()
        plt.show()
    
    def plot_timing_comparison(self):
        """Plot training and prediction time comparison"""
        if not self.results:
            print("No results to plot")
            return
        
        # Aggregate timing data
        models = list(self.models.keys())
        train_times = []
        predict_times = []
        
        for model in models:
            total_train = sum(self.results[dataset][model]['train_time'] 
                            for dataset in self.results.keys())
            total_predict = sum(self.results[dataset][model]['predict_time'] 
                              for dataset in self.results.keys())
            train_times.append(total_train)
            predict_times.append(total_predict)
        
        # Plot
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, train_times, width, label='Training Time', alpha=0.8)
        bars2 = ax.bar(x + width/2, predict_times, width, label='Prediction Time', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training vs Prediction Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\nCOMPREHENSIVE MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        # Overall performance summary
        models = list(self.models.keys())
        datasets = list(self.results.keys())
        
        print(f"\nDatasets analyzed: {len(datasets)}")
        print(f"Models compared: {len(models)}")
        
        # Calculate average performance across all datasets
        avg_performance = {}
        for model in models:
            total_accuracy = sum(self.results[dataset][model]['accuracy'] 
                               for dataset in datasets)
            avg_accuracy = total_accuracy / len(datasets)
            avg_performance[model] = avg_accuracy
        
        print(f"\nAverage Accuracy Across All Datasets:")
        for model, acc in sorted(avg_performance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:20}: {acc:.4f}")
        
        # Best performing model per dataset
        print(f"\nBest Performing Model per Dataset:")
        for dataset in datasets:
            best_model = max(self.results[dataset].items(), 
                           key=lambda x: x[1]['accuracy'])
            print(f"  {dataset:20}: {best_model[0]} ({best_model[1]['accuracy']:.4f})")
        
        # k-NN specific analysis
        print(f"\nk-NN Specific Analysis:")
        knn_results = [self.results[dataset]['k-NN'] for dataset in datasets]
        avg_knn_accuracy = np.mean([r['accuracy'] for r in knn_results])
        avg_knn_cv = np.mean([r['cv_mean'] for r in knn_results])
        avg_train_time = np.mean([r['train_time'] for r in knn_results])
        avg_predict_time = np.mean([r['predict_time'] for r in knn_results])
        
        print(f"  Average Accuracy: {avg_knn_accuracy:.4f}")
        print(f"  Average CV Score: {avg_knn_cv:.4f}")
        print(f"  Average Training Time: {avg_train_time:.4f}s")
        print(f"  Average Prediction Time: {avg_predict_time:.4f}s")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print(f"  • k-NN works best on: {datasets[np.argmax([self.results[d]['k-NN']['accuracy'] for d in datasets])]}")
        print(f"  • Overall best model: {max(avg_performance.items(), key=lambda x: x[1])[0]}")
        print(f"  • k-NN is {'competitive' if avg_knn_accuracy >= sorted(avg_performance.values())[-2] else 'less competitive'} compared to other models")


def optimize_knn_hyperparameters(X_train, y_train, X_test, y_test):
    """
    Find optimal hyperparameters for k-NN using Grid Search
    
    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    best_score : float
        Best cross-validation score
    """
    print("Optimizing k-NN hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Grid search with cross-validation
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Test best model
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy with best params: {test_accuracy:.4f}")
    
    return grid_search.best_params_, grid_search.best_score_, test_accuracy


def analyze_knn_distance_metrics(X_train, y_train, X_test, y_test):
    """Detailed analysis of different distance metrics for k-NN"""
    print("\nDetailed Distance Metric Analysis:")
    print("=" * 40)
    
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    k_values = [1, 3, 5, 7, 9, 11]
    
    results = {}
    
    for metric in distance_metrics:
        metric_results = []
        for k in k_values:
            try:
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                metric_results.append(accuracy)
            except Exception as e:
                print(f"Error with {metric} and k={k}: {e}")
                metric_results.append(0)
        
        results[metric] = metric_results
    
    # Plot results
    plt.figure(figsize=(12, 8))
    for metric, accuracies in results.items():
        plt.plot(k_values, accuracies, marker='o', linewidth=2, label=metric.capitalize())
    
    plt.title('k-NN Performance by Distance Metric and k Value')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """Main function to run comprehensive model comparison"""
    print("Comprehensive Machine Learning Model Comparison")
    print("Focus: k-NN vs Other Popular Algorithms")
    print("=" * 55)
    
    # Import dataset preparation functions
    from dataset_downloader import main as download_datasets, prepare_dataset_for_knn
    from knn_implementation import load_and_prepare_data
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load built-in datasets
    print("\n1. Loading built-in datasets...")
    X_train, X_test, y_train, y_test, _, _ = load_and_prepare_data()
    comparator.evaluate_models(X_train, X_test, y_train, y_test, "Iris")
    
    # Try to load additional datasets
    try:
        print("\n2. Loading additional datasets...")
        datasets = download_datasets()
        
        for name, df in datasets.items():
            if name == 'wine_quality':
                target_col = 'quality'
            elif name == 'heart_disease':
                target_col = 'target'
            elif name == 'diabetes':
                target_col = 'outcome'
            else:
                target_col = 'target'
            
            try:
                X_train, X_test, y_train, y_test, _ = prepare_dataset_for_knn(df, target_col)
                comparator.evaluate_models(X_train, X_test, y_train, y_test, name.replace('_', ' ').title())
            except Exception as e:
                print(f"Error processing {name}: {e}")
    
    except Exception as e:
        print(f"Could not load additional datasets: {e}")
    
    # Generate visualizations and reports
    print("\n3. Generating analysis results...")
    
    # Plot comparisons for each dataset
    for dataset_name in comparator.results.keys():
        comparator.plot_model_comparison(dataset_name, 'accuracy')
    
    # Overall performance matrix
    comparator.plot_performance_matrix()
    
    # Timing comparison
    comparator.plot_timing_comparison()
    
    # Generate comprehensive report
    comparator.generate_summary_report()
    
    # k-NN specific optimizations
    print("\n4. k-NN Hyperparameter Optimization:")
    X_train, X_test, y_train, y_test, _, _ = load_and_prepare_data()
    optimize_knn_hyperparameters(X_train, y_train, X_test, y_test)
    
    # Distance metrics analysis
    analyze_knn_distance_metrics(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

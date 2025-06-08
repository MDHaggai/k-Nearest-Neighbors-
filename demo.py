"""
Quick Demo Script for k-NN Implementation
This script provides a quick demonstration of the k-NN algorithm with visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add src directory to path
sys.path.append('src')
from knn_implementation import KNearestNeighbors

def create_2d_visualization_dataset():
    """Create a simple 2D dataset for visualization"""
    np.random.seed(42)
    
    # Create three clusters
    cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 50)
    cluster2 = np.random.multivariate_normal([6, 6], [[0.5, 0], [0, 0.5]], 50)
    cluster3 = np.random.multivariate_normal([2, 6], [[0.5, 0], [0, 0.5]], 50)
    
    X = np.vstack([cluster1, cluster2, cluster3])
    y = np.array([0]*50 + [1]*50 + [2]*50)
    
    return X, y

def visualize_knn_decision_boundary():
    """Visualize k-NN decision boundary"""
    print("Creating 2D visualization of k-NN decision boundary...")
    
    # Create dataset
    X, y = create_2d_visualization_dataset()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train k-NN
    knn = KNearestNeighbors(k=5)
    knn.fit(X_train, y_train)
    
    # Create a mesh for decision boundary
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot decision boundary
    plt.subplot(2, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('k-NN Decision Boundary (k=5)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    # Plot training vs test points
    plt.subplot(2, 2, 2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', 
                s=50, cmap=plt.cm.RdYlBu, alpha=0.7, label='Training')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^', 
                s=80, cmap=plt.cm.RdYlBu, edgecolors='black', label='Test')
    plt.title('Training vs Test Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Test different k values
    k_values = [1, 3, 5, 10, 15]
    accuracies = []
    
    for k in k_values:
        knn_k = KNearestNeighbors(k=k)
        knn_k.fit(X_train, y_train)
        y_pred = knn_k.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    plt.subplot(2, 2, 3)
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
    plt.title('Accuracy vs k Value')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Compare distance metrics
    metrics = ['euclidean', 'manhattan', 'cosine']
    metric_acc = []
    
    for metric in metrics:
        knn_metric = KNearestNeighbors(k=5, distance_metric=metric)
        knn_metric.fit(X_train, y_train)
        y_pred = knn_metric.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        metric_acc.append(acc)
    
    plt.subplot(2, 2, 4)
    bars = plt.bar(metrics, metric_acc, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Distance Metrics Comparison')
    plt.ylabel('Accuracy')
    
    # Add value labels on bars
    for bar, acc in zip(bars, metric_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/knn_visualization_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Best k value: {k_values[np.argmax(accuracies)]} with accuracy: {max(accuracies):.4f}")
    print(f"Best distance metric: {metrics[np.argmax(metric_acc)]} with accuracy: {max(metric_acc):.4f}")

def quick_iris_demo():
    """Quick demonstration on Iris dataset"""
    print("\nRunning quick Iris dataset demonstration...")
    
    # Load and prepare Iris data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test our custom k-NN
    knn_custom = KNearestNeighbors(k=5)
    knn_custom.fit(X_train_scaled, y_train)
    y_pred_custom = knn_custom.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred_custom)
    
    print(f"Custom k-NN Accuracy on Iris: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_custom, target_names=iris.target_names))

def demonstrate_curse_of_dimensionality():
    """Demonstrate the curse of dimensionality effect"""
    print("\nDemonstrating Curse of Dimensionality...")
    
    n_samples = 500
    dimensions = [2, 5, 10, 20, 50, 100]
    accuracies = []
    
    for dim in dimensions:
        # Create synthetic dataset
        X, y = make_classification(n_samples=n_samples, n_features=dim, 
                                  n_informative=min(dim, 10), n_redundant=0,
                                  n_clusters_per_class=1, random_state=42)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test k-NN
        knn = KNearestNeighbors(k=5)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"Dimensions: {dim:3d}, Accuracy: {acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, accuracies, marker='o', linewidth=2, markersize=8, color='red')
    plt.title('k-NN Performance vs Dimensionality (Curse of Dimensionality)')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Chance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/curse_of_dimensionality.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main demo function"""
    print("k-NN Implementation Demo")
    print("=" * 30)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run demonstrations
    visualize_knn_decision_boundary()
    quick_iris_demo()
    demonstrate_curse_of_dimensionality()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the 'results' folder for saved plots.")
    print("Next steps:")
    print("1. Run 'python src/knn_implementation.py' for full analysis")
    print("2. Run 'python src/dataset_downloader.py' to get real datasets") 
    print("3. Run 'python src/model_comparison.py' for model comparison")
    print("4. Open 'notebooks/knn_analysis.ipynb' for interactive analysis")

if __name__ == "__main__":
    main()

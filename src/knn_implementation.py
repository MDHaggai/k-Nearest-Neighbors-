"""
k-Nearest Neighbors (k-NN) Implementation
This module implements a basic k-NN classifier from scratch and includes
utilities for distance calculation and performance evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


class KNearestNeighbors:
    """
    k-Nearest Neighbors Classifier Implementation
    
    This is a lazy learning algorithm that makes predictions based on the 
    k closest training examples in the feature space.
    
    Parameters:
    -----------
    k : int, default=3
        Number of neighbors to use for prediction
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'cosine')
    """
    
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        """Calculate Manhattan distance between two points"""
        return np.sum(np.abs(x1 - x2))
    
    def cosine_distance(self, x1, x2):
        """Calculate Cosine distance between two points"""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - (dot_product / (norm_x1 * norm_x2))
    
    def calculate_distance(self, x1, x2):
        """Calculate distance based on chosen metric"""
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(x1, x2)
        else:
            raise ValueError("Unsupported distance metric")
    
    def fit(self, X, y):
        """
        Fit the k-NN model (lazy learning - just store the training data)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict_single(self, x):
        """Predict class for a single sample"""
        # Calculate distances to all training points
        distances = []
        for i, x_train in enumerate(self.X_train):
            distance = self.calculate_distance(x, x_train)
            distances.append((distance, self.y_train[i]))
        
        # Sort by distance and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Get the labels of k nearest neighbors
        k_nearest_labels = [label for _, label in k_nearest]
        
        # Return the most common class
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """
        Predict classes for multiple samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return np.array(predictions)


def load_and_prepare_data():
    """
    Load and prepare the Iris dataset for k-NN classification
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split and preprocessed data
    feature_names : list
        Names of the features
    target_names : list
        Names of the target classes
    """
    from sklearn.datasets import load_iris
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print("Dataset Information:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {iris.feature_names}")
    print(f"Target names: {iris.target_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, iris.feature_names, iris.target_names


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    model_name : str
        Name of the model for display
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return accuracy


def plot_confusion_matrix(y_true, y_pred, target_names, model_name="Model"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def compare_k_values(X_train, X_test, y_train, y_test):
    """Compare performance across different k values"""
    k_values = range(1, 21)
    accuracies = []
    
    for k in k_values:
        knn = KNearestNeighbors(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=6)
    plt.title('k-NN Performance vs. k Value')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    best_k = k_values[np.argmax(accuracies)]
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Best k = {best_k}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return best_k, max(accuracies)


def compare_distance_metrics(X_train, X_test, y_train, y_test, k=5):
    """Compare different distance metrics"""
    metrics = ['euclidean', 'manhattan', 'cosine']
    results = {}
    
    print("\nComparing Distance Metrics:")
    print("=" * 40)
    
    for metric in metrics:
        knn = KNearestNeighbors(k=k, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[metric] = accuracy
        print(f"{metric.capitalize()} Distance: {accuracy:.4f}")
    
    return results


def main():
    """Main function to run the complete k-NN analysis"""
    print("k-Nearest Neighbors (k-NN) Classification Analysis")
    print("=" * 55)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data()
    
    # 1. Basic k-NN implementation
    print("\n1. Basic k-NN Implementation (k=5)")
    knn_custom = KNearestNeighbors(k=5)
    knn_custom.fit(X_train, y_train)
    y_pred_custom = knn_custom.predict(X_test)
    accuracy_custom = evaluate_model(y_test, y_pred_custom, "Custom k-NN")
    
    # 2. Compare with sklearn implementation
    print("\n2. Scikit-learn k-NN Implementation (k=5)")
    knn_sklearn = KNeighborsClassifier(n_neighbors=5)
    knn_sklearn.fit(X_train, y_train)
    y_pred_sklearn = knn_sklearn.predict(X_test)
    accuracy_sklearn = evaluate_model(y_test, y_pred_sklearn, "Scikit-learn k-NN")
    
    # 3. Plot confusion matrices
    plot_confusion_matrix(y_test, y_pred_custom, target_names, "Custom k-NN")
    plot_confusion_matrix(y_test, y_pred_sklearn, target_names, "Scikit-learn k-NN")
    
    # 4. Find optimal k value
    print("\n3. Finding Optimal k Value")
    best_k, best_accuracy = compare_k_values(X_train, X_test, y_train, y_test)
    print(f"Best k value: {best_k} with accuracy: {best_accuracy:.4f}")
    
    # 5. Compare distance metrics
    print("\n4. Comparing Distance Metrics")
    metric_results = compare_distance_metrics(X_train, X_test, y_train, y_test)
    
    # 6. Performance summary
    print("\n" + "=" * 55)
    print("PERFORMANCE SUMMARY")
    print("=" * 55)
    print(f"Custom k-NN Accuracy:      {accuracy_custom:.4f}")
    print(f"Scikit-learn k-NN Accuracy: {accuracy_sklearn:.4f}")
    print(f"Best k value:              {best_k}")
    print(f"Best accuracy with optimal k: {best_accuracy:.4f}")
    print("\nDistance Metric Comparison:")
    for metric, acc in metric_results.items():
        print(f"  {metric.capitalize()}: {acc:.4f}")


if __name__ == "__main__":
    main()

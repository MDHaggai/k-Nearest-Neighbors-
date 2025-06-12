"""
Complete Demo Script for k-NN Banknote Authentication Project
=============================================================

This script demonstrates the entire project workflow:
1. Educational notebooks for k-NN theory and practice
2. Model training and evaluation
3. Beautiful GUI application for real-time banknote authentication

Run this script to see all components in action.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def print_banner(text):
    """Print a beautiful banner"""
    print("\n" + "="*60)
    print(f"🎯 {text}")
    print("="*60)

def print_step(step_num, description):
    """Print a step description"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 40)

def check_requirements():
    """Check if all requirements are installed"""
    print_step(1, "Checking Requirements")
    
    required_modules = [
        'tkinter', 'PIL', 'cv2', 'sklearn', 'numpy', 
        'pandas', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing modules: {', '.join(missing)}")
        print("Run: pip install -r gui_requirements.txt")
        return False
    
    print("\n🎉 All requirements satisfied!")
    return True

def show_project_structure():
    """Display the project structure"""
    print_step(2, "Project Structure")
    
    print("📁 k-Nearest-Neighbors/")
    print("├── 📊 Educational Notebooks:")
    print("│   ├── 01_data_loading_and_knn_theory.ipynb")
    print("│   ├── 02_training_and_lazy_learning.ipynb") 
    print("│   ├── 03_distance_metrics_and_performance.ipynb")
    print("│   ├── 04_high_dimensional_analysis.ipynb")
    print("│   └── knn_analysis.ipynb (comprehensive)")
    print("│")
    print("├── 🤖 Machine Learning Implementation:")
    print("│   ├── src/knn_implementation.py")
    print("│   ├── src/model_comparison.py")
    print("│   └── src/dataset_downloader.py")
    print("│")
    print("├── 🖥️  GUI Application:")
    print("│   ├── gui_banknote_classifier.py")
    print("│   ├── advanced_image_processing.py")
    print("│   └── generate_sample_images.py")
    print("│")
    print("├── 📈 Data & Results:")
    print("│   ├── data/banknote_authentication.csv")
    print("│   ├── sample_images/ (12 demo images)")
    print("│   └── results/ (analysis outputs)")
    print("│")
    print("└── 📚 Documentation:")
    print("    ├── README.md (main documentation)")
    print("    ├── GUI_README.md (GUI guide)")
    print("    └── requirements files")

def show_dataset_info():
    """Show dataset information"""
    print_step(3, "Dataset Information")
    
    try:
        import pandas as pd
        df = pd.read_csv('data/banknote_authentication.csv')
        
        print(f"📊 Banknote Authentication Dataset:")
        print(f"   • Total samples: {len(df):,}")
        print(f"   • Features: {len(df.columns)-1}")
        print(f"   • Classes: {df['class'].nunique()}")
        print(f"   • Genuine banknotes: {(df['class'] == 0).sum():,}")
        print(f"   • Forged banknotes: {(df['class'] == 1).sum():,}")
        
        print(f"\n🔬 Feature Statistics:")
        for col in ['variance', 'skewness', 'curtosis', 'entropy']:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"   • {col.capitalize()}: μ={mean_val:.3f}, σ={std_val:.3f}")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

def demonstrate_model_performance():
    """Demonstrate model performance"""
    print_step(4, "Model Performance Demo")
    
    try:
        sys.path.append('src')
        from knn_implementation import KNearestNeighbors
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
        import pandas as pd
        import numpy as np
        
        # Load data
        df = pd.read_csv('data/banknote_authentication.csv')
        X = df[['variance', 'skewness', 'curtosis', 'entropy']].values
        y = df['class'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("🔄 Training k-NN model...")
        knn = KNearestNeighbors(k=5)
        knn.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"🎯 Test Accuracy: {accuracy:.2%}")
        print(f"📊 Training samples: {len(X_train):,}")
        print(f"🧪 Test samples: {len(X_test):,}")
        
        # Show sample predictions
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(5, len(X_test))):
            actual = "Genuine" if y_test[i] == 0 else "Forged"
            predicted = "Genuine" if y_pred[i] == 0 else "Forged"
            status = "✅" if y_test[i] == y_pred[i] else "❌"
            print(f"   {status} Actual: {actual}, Predicted: {predicted}")
            
    except Exception as e:
        print(f"❌ Error in model demo: {e}")

def show_sample_images():
    """Show information about sample images"""
    print_step(5, "Sample Images")
    
    sample_dir = Path('sample_images')
    if sample_dir.exists():
        genuine_images = list(sample_dir.glob('genuine_*.jpg'))
        forged_images = list(sample_dir.glob('forged_*.jpg'))
        
        print(f"🖼️  Generated {len(genuine_images) + len(forged_images)} sample images:")
        print(f"   • {len(genuine_images)} genuine banknote images")
        print(f"   • {len(forged_images)} forged banknote images")
        print(f"   • Location: {sample_dir.absolute()}")
        
        print(f"\n📝 Image characteristics:")
        print(f"   • Genuine: Consistent patterns, minimal noise")
        print(f"   • Forged: Irregular patterns, printing artifacts")
        print(f"   • Format: JPEG, 400x200 pixels")
        print(f"   • Purpose: GUI demonstration and testing")
    else:
        print(f"❌ Sample images not found. Run: python generate_sample_images.py")

def launch_gui():
    """Launch the GUI application"""
    print_step(6, "Launching GUI Application")
    
    print("🚀 Starting Banknote Authentication GUI...")
    print("📱 Features:")
    print("   • Beautiful dark theme interface")
    print("   • Real-time image processing")
    print("   • k-NN classification with confidence scoring")
    print("   • Feature extraction visualization")
    print("   • Sample image testing capability")
    
    print(f"\n💡 Usage:")
    print(f"   1. Click 'Choose Image File' to upload a banknote image")
    print(f"   2. Or click 'Use Sample Images' to try demo images")
    print(f"   3. View real-time authentication results")
    print(f"   4. Examine extracted features and confidence scores")
    
    print(f"\n⏳ Launching GUI in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    try:
        # Launch GUI
        subprocess.run([sys.executable, 'gui_banknote_classifier.py'], 
                      cwd=os.getcwd())
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        print(f"💡 Try running manually: python gui_banknote_classifier.py")

def show_educational_resources():
    """Show educational resources"""
    print_step(7, "Educational Resources")
    
    print("📚 Available Jupyter Notebooks:")
    
    notebooks = [
        ("01_data_loading_and_knn_theory.ipynb", 
         "k-NN theory, lazy learning concepts, distance metrics"),
        ("02_training_and_lazy_learning.ipynb", 
         "Training process, lazy vs eager learning, timing analysis"),
        ("03_distance_metrics_and_performance.ipynb", 
         "Distance metrics comparison, hyperparameter tuning"),
        ("04_high_dimensional_analysis.ipynb", 
         "Curse of dimensionality, PCA, feature selection"),
        ("knn_analysis.ipynb", 
         "Comprehensive analysis with all datasets")
    ]
    
    for i, (notebook, description) in enumerate(notebooks, 1):
        print(f"   {i}. {notebook}")
        print(f"      {description}")
    
    print(f"\n🎓 Learning Path:")
    print(f"   1. Start with theory notebooks (01-04)")
    print(f"   2. Run the comprehensive analysis (knn_analysis.ipynb)")
    print(f"   3. Try the GUI application")
    print(f"   4. Experiment with your own images")

def main():
    """Main demo function"""
    print_banner("k-NN Banknote Authentication Project Demo")
    
    print("🎯 This project demonstrates:")
    print("   • Educational k-NN implementation and analysis")
    print("   • Real-world machine learning application")
    print("   • Beautiful GUI for practical use")
    print("   • Comprehensive documentation and examples")
    
    # Run all demo steps
    if not check_requirements():
        return
    
    show_project_structure()
    show_dataset_info()
    demonstrate_model_performance()
    show_sample_images()
    show_educational_resources()
    
    # Ask user if they want to launch GUI
    print(f"\n" + "="*60)
    print("🎯 Demo Complete!")
    print("="*60)
    
    response = input("\n🚀 Would you like to launch the GUI application? (y/n): ")
    if response.lower() in ['y', 'yes']:
        launch_gui()
    else:
        print("\n💡 To launch GUI manually, run: python gui_banknote_classifier.py")
        print("📚 To explore notebooks, run: jupyter notebook notebooks/")
    
    print(f"\n🙏 Thank you for exploring the k-NN Banknote Authentication project!")
    print(f"⭐ If you found this useful, please star the repository!")

if __name__ == "__main__":
    main()

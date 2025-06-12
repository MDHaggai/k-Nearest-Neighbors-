"""
Advanced Image Processing Module for Banknote Authentication
============================================================

This module provides sophisticated image processing functions to extract
meaningful features from banknote images that correlate with the dataset features.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import feature, filters, measure, segmentation
from sklearn.preprocessing import StandardScaler
import pywt


class BanknoteImageProcessor:
    """Advanced image processing for banknote feature extraction"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
        # Define feature extraction parameters
        self.image_size = (256, 128)  # Standard processing size
        self.wavelet = 'db4'  # Daubechies wavelet for texture analysis
        
    def preprocess_image(self, image_path):
        """Preprocess banknote image for feature extraction"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            resized = cv2.resize(gray, self.image_size)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            
            # Reduce noise with bilateral filter
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Normalize to [0, 1]
            normalized = denoised.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_wavelet_features(self, image):
        """Extract wavelet-based features similar to the dataset"""
        try:
            # Perform 2D wavelet decomposition
            coeffs = pywt.dwt2(image, self.wavelet)
            cA, (cH, cV, cD) = coeffs
            
            # Combine all detail coefficients
            details = np.concatenate([cH.flatten(), cV.flatten(), cD.flatten()])
            
            # Calculate statistical moments
            
            # Feature 1: Variance of wavelet coefficients
            variance = np.var(details)
            
            # Feature 2: Skewness
            mean_val = np.mean(details)
            std_val = np.std(details)
            if std_val > 0:
                skewness = np.mean(((details - mean_val) / std_val) ** 3)
            else:
                skewness = 0
            
            # Feature 3: Kurtosis
            if std_val > 0:
                kurtosis = np.mean(((details - mean_val) / std_val) ** 4) - 3
            else:
                kurtosis = 0
            
            # Feature 4: Entropy of the approximation coefficients
            # Calculate histogram and entropy
            hist, _ = np.histogram(cA.flatten(), bins=64, density=True)
            hist = hist + 1e-10  # Avoid log(0)
            entropy = -np.sum(hist * np.log2(hist))
            
            return np.array([variance, skewness, kurtosis, entropy])
            
        except Exception as e:
            print(f"Error extracting wavelet features: {e}")
            return None
    
    def extract_texture_features(self, image):
        """Extract additional texture features for enhanced analysis"""
        try:
            # Local Binary Pattern (LBP)
            lbp = feature.local_binary_pattern(image, 24, 3, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, density=True)
            
            # Gray Level Co-occurrence Matrix (GLCM) properties
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            # Convert to uint8 for GLCM
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Calculate GLCM properties
            contrast_values = []
            dissimilarity_values = []
            homogeneity_values = []
            energy_values = []
            
            for distance in distances:
                for angle in angles:
                    glcm = feature.graycomatrix(image_uint8, [distance], [angle], 
                                              levels=256, symmetric=True, normed=True)
                    
                    contrast_values.append(feature.graycoprops(glcm, 'contrast')[0, 0])
                    dissimilarity_values.append(feature.graycoprops(glcm, 'dissimilarity')[0, 0])
                    homogeneity_values.append(feature.graycoprops(glcm, 'homogeneity')[0, 0])
                    energy_values.append(feature.graycoprops(glcm, 'energy')[0, 0])
            
            # Aggregate GLCM features
            glcm_features = np.array([
                np.mean(contrast_values),
                np.mean(dissimilarity_values),
                np.mean(homogeneity_values),
                np.mean(energy_values)
            ])
            
            # Gabor filter responses
            gabor_responses = []
            for frequency in [0.1, 0.3, 0.5]:
                for angle in [0, 45, 90, 135]:
                    real, _ = filters.gabor(image, frequency=frequency, 
                                          theta=np.deg2rad(angle))
                    gabor_responses.append(np.std(real))
            
            gabor_features = np.array(gabor_responses)
            
            return {
                'lbp_histogram': lbp_hist,
                'glcm_features': glcm_features,
                'gabor_features': gabor_features
            }
            
        except Exception as e:
            print(f"Error extracting texture features: {e}")
            return None
    
    def extract_geometric_features(self, image):
        """Extract geometric and structural features"""
        try:
            # Edge detection
            edges = feature.canny(image, sigma=1, low_threshold=0.1, high_threshold=0.2)
            
            # Calculate edge density
            edge_density = np.sum(edges) / edges.size
            
            # Hough line detection for structural analysis
            lines = cv2.HoughLinesP((edges * 255).astype(np.uint8), 
                                   1, np.pi/180, threshold=30, 
                                   minLineLength=20, maxLineGap=10)
            
            line_count = len(lines) if lines is not None else 0
            
            # Calculate average line length
            if lines is not None and len(lines) > 0:
                line_lengths = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    line_lengths.append(length)
                avg_line_length = np.mean(line_lengths)
            else:
                avg_line_length = 0
            
            # Corner detection
            corners = feature.corner_harris(image)
            corner_count = len(feature.corner_peaks(corners, min_distance=5))
            
            return np.array([edge_density, line_count, avg_line_length, corner_count])
            
        except Exception as e:
            print(f"Error extracting geometric features: {e}")
            return None
    
    def extract_all_features(self, image_path):
        """Extract comprehensive feature set from banknote image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return None
            
            # Extract primary wavelet features (matching dataset)
            wavelet_features = self.extract_wavelet_features(processed_image)
            
            # Extract additional features for analysis
            texture_features = self.extract_texture_features(processed_image)
            geometric_features = self.extract_geometric_features(processed_image)
            
            if wavelet_features is None:
                return None
            
            feature_dict = {
                'primary_features': wavelet_features,  # These match the training data
                'texture_features': texture_features,
                'geometric_features': geometric_features,
                'processed_image': processed_image
            }
            
            return feature_dict
            
        except Exception as e:
            print(f"Error in feature extraction pipeline: {e}")
            return None
    
    def analyze_image_quality(self, image_path):
        """Analyze image quality metrics"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return None
            
            # Calculate various quality metrics
            
            # Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(image, cv2.CV_64F).var()
            
            # Brightness analysis
            brightness = np.mean(image)
            
            # Contrast analysis
            contrast = np.std(image)
            
            # Noise estimation
            noise_estimate = np.std(image - filters.gaussian(image, sigma=1))
            
            # Resolution adequacy (based on edge sharpness)
            edges = feature.canny(image, sigma=1)
            edge_sharpness = np.sum(edges) / edges.size
            
            quality_metrics = {
                'blur_score': blur_score,
                'brightness': brightness,
                'contrast': contrast,
                'noise_level': noise_estimate,
                'edge_sharpness': edge_sharpness,
                'overall_quality': min(blur_score/100, contrast*10, edge_sharpness*1000)
            }
            
            return quality_metrics
            
        except Exception as e:
            print(f"Error analyzing image quality: {e}")
            return None


# Additional utility functions for the GUI

def create_feature_visualization(features_dict):
    """Create visualizations of extracted features"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if features_dict is None:
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Banknote Feature Analysis', fontsize=16, fontweight='bold')
    
    # Primary features (wavelet-based)
    primary_features = features_dict['primary_features']
    feature_names = ['Variance', 'Skewness', 'Kurtosis', 'Entropy']
    
    # Bar plot of primary features
    axes[0, 0].bar(feature_names, primary_features, color=['#89b4f0', '#a6e3a1', '#f9e2af', '#f38ba8'])
    axes[0, 0].set_title('Primary Features (Wavelet-based)')
    axes[0, 0].set_ylabel('Feature Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Processed image
    if 'processed_image' in features_dict:
        axes[0, 1].imshow(features_dict['processed_image'], cmap='gray')
        axes[0, 1].set_title('Processed Image')
        axes[0, 1].axis('off')
    
    # Texture features visualization
    if features_dict['texture_features'] is not None:
        texture_data = features_dict['texture_features']
        
        # GLCM features
        glcm_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy']
        axes[1, 0].bar(glcm_names, texture_data['glcm_features'], 
                       color='lightblue', alpha=0.7)
        axes[1, 0].set_title('GLCM Texture Features')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # LBP histogram
        axes[1, 1].plot(texture_data['lbp_histogram'], color='orange', linewidth=2)
        axes[1, 1].set_title('Local Binary Pattern Histogram')
        axes[1, 1].set_xlabel('LBP Bin')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig


def estimate_authenticity_confidence(features, model, scaler):
    """Estimate authenticity with detailed confidence analysis"""
    try:
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Get prediction and probabilities
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get k nearest neighbors for analysis
        distances, indices = model.kneighbors(features_scaled)
        
        # Calculate various confidence metrics
        confidence_metrics = {
            'primary_confidence': np.max(probabilities),
            'prediction': prediction,
            'probabilities': probabilities,
            'nearest_distances': distances[0],
            'class_distribution': np.bincount(model._y[indices[0]]),
            'decision_margin': abs(probabilities[1] - probabilities[0])
        }
        
        return confidence_metrics
        
    except Exception as e:
        print(f"Error in confidence estimation: {e}")
        return None

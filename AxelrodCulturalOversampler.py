import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter

class AxelrodCulturalOversampler:

    
    def __init__(self, k_neighbors,cultural_traits, similarity_threshold, influence_rate, 
                 sampling_strategy='auto', random_state=None):

        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.cultural_traits = cultural_traits
        self.similarity_threshold = similarity_threshold
        self.influence_rate = influence_rate
        self.rng = np.random.RandomState(self.random_state)
        
    def fit_resample(self, X, y):
        # Convert inputs to numpy arrays
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        
        # Count samples per class
        class_counts = Counter(y_array)
        majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
        majority_count = class_counts[majority_class]
        
        # Determine sampling strategy
        if self.sampling_strategy == 'auto':
            sampling_strategy = {cls: majority_count for cls in class_counts.keys() 
                               if cls != majority_class}
        elif isinstance(self.sampling_strategy, dict):
            sampling_strategy = self.sampling_strategy
        else:
            raise ValueError("sampling_strategy must be 'auto' or a dictionary")
        
        # Initialize output arrays with existing samples
        X_resampled = X_array.copy()
        y_resampled = y_array.copy()
        
        # Group feature indices into cultural traits
        n_features = X_array.shape[1]
        features_per_trait = max(1, n_features // self.cultural_traits)
        trait_indices = [list(range(i, min(i+features_per_trait, n_features))) 
                        for i in range(0, n_features, features_per_trait)]
        
        # For each class that needs oversampling
        for class_label, n_samples in sampling_strategy.items():
            if n_samples <= class_counts[class_label]:
                continue  # Skip if no oversampling needed
            
            # Get samples for this class only
            class_indices = np.where(y_array == class_label)[0]
            class_samples = X_array[class_indices]
            
            # Fit nearest neighbors model on the class samples only
            n_neighbors = min(self.k_neighbors + 1, len(class_samples))
            if n_neighbors <= 1:
                print(f"Warning: Class {class_label} has only {len(class_samples)} samples. Using SMOTE-like approach.")
                continue
                
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(class_samples)
            
            n_samples_to_generate = n_samples - class_counts[class_label]
            synthetic_samples = []
            
            for _ in range(n_samples_to_generate):
                # Randomly select a sample as the base
                base_idx = self.rng.randint(0, len(class_samples))
                base_sample = class_samples[base_idx]
                
                # Find its nearest same-class neighbors
                distances, indices = nn.kneighbors([base_sample], n_neighbors=n_neighbors)
                # Remove the sample itself from neighbors
                neighbor_indices = indices[0][1:]
                
                # Create new sample starting with base sample
                new_sample = base_sample.copy()
                
                # For each cultural trait (feature group)
                for trait_idx in trait_indices:
                    # For each potential neighbor for cultural exchange
                    # Randomly choose neighbors to consider (for efficiency)
                    neighbors_to_consider = self.rng.choice(
                        neighbor_indices, 
                        size=min(3, len(neighbor_indices)), 
                        replace=False
                    )
                    
                    for local_idx in neighbors_to_consider:
                        neighbor = class_samples[local_idx]
                        
                        # Calculate cultural similarity for this trait between base and neighbor
                        trait_similarity = 1 - np.mean(np.abs(base_sample[trait_idx] - neighbor[trait_idx]))
                        
                        # Cultural exchange occurs with probability proportional to similarity
                        if trait_similarity > self.similarity_threshold and self.rng.random() < self.influence_rate:
                            # Adopt trait (feature values) from neighbor with some variation
                            blend_ratio = self.rng.beta(2, 2)  # Creates more realistic blending
                            new_sample[trait_idx] = (blend_ratio * base_sample[trait_idx] + 
                                                    (1 - blend_ratio) * neighbor[trait_idx])
                            
                            # Sometimes add small variation for diversity
                            if self.rng.random() < 0.3:
                                variation = self.rng.normal(0, 0.1, size=len(trait_idx))
                                # Scale variation by feature range
                                feature_ranges = np.max(class_samples[:, trait_idx], axis=0) - np.min(class_samples[:, trait_idx], axis=0)
                                # Avoid division by zero by adding small epsilon
                                feature_ranges = np.maximum(feature_ranges, 1e-8)
                                scaled_variation = variation * feature_ranges * 0.05
                                new_sample[trait_idx] += scaled_variation
                
                synthetic_samples.append(new_sample)
            
            # Add synthetic samples to output
            X_resampled = np.vstack([X_resampled] + synthetic_samples)
            y_resampled = np.hstack([y_resampled] + [class_label] * len(synthetic_samples))
        
        return X_resampled, y_resampled
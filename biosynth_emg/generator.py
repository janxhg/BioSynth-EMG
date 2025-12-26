import numpy as np
import h5py
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time
from tqdm import tqdm

from .biomechanical import BiomechanicalLayer
from .electrical import ElectricalLayer


class BioSynthGenerator:
    """
    Main generator class for BioSynth-EMG synthetic signals.
    Combines biomechanical and electrical layers to generate realistic EMG datasets.
    """
    
    def __init__(self, sampling_rate: int = 2000, num_motor_units: int = 100, 
                 num_channels: int = 8, random_seed: Optional[int] = None):
        """
        Initialize the BioSynth-EMG generator.
        
        Args:
            sampling_rate: Sampling rate in Hz (default 2000)
            num_motor_units: Number of motor units to simulate
            num_channels: Number of EMG channels (8 for Myo armband)
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.sampling_rate = sampling_rate
        self.num_motor_units = num_motor_units
        self.num_channels = num_channels
        
        # Initialize layers
        self.biomechanical = BiomechanicalLayer(sampling_rate, num_motor_units)
        self.electrical = ElectricalLayer(sampling_rate, num_channels)
        
        # Gesture definitions
        self.gestures = {
            0: 'rest',
            1: 'fist', 
            2: 'extension',
            3: 'flexion',
            4: 'supination',
            5: 'pronation',
            6: 'radial_deviation',
            7: 'ulnar_deviation'
        }
        
        # Gesture-specific activation patterns
        self.gesture_activations = self._define_gesture_patterns()
    
    def _define_gesture_patterns(self) -> Dict[int, Dict[str, float]]:
        """Define muscle activation patterns for each gesture."""
        patterns = {
            0: {'activation': 0.05, 'duration': 1.0},  # rest
            1: {'activation': 0.8, 'duration': 2.0},   # fist
            2: {'activation': 0.6, 'duration': 1.5},   # extension
            3: {'activation': 0.7, 'duration': 1.5},   # flexion
            4: {'activation': 0.5, 'duration': 1.2},   # supination
            5: {'activation': 0.5, 'duration': 1.2},   # pronation
            6: {'activation': 0.4, 'duration': 1.0},   # radial deviation
            7: {'activation': 0.4, 'duration': 1.0}    # ulnar deviation
        }
        return patterns
    
    def generate_single_sample(self, gesture_id: int, force_level: float = 0.5,
                             duration: float = 1.0, fatigue_factor: float = 1.0,
                             tissue_depth: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Generate a single EMG sample with specified parameters.
        
        Args:
            gesture_id: Gesture ID (0-7)
            force_level: Force level (0.0-1.0)
            duration: Duration in seconds
            fatigue_factor: Fatigue reduction factor (0-1)
            tissue_depth: Optional tissue depth override
            
        Returns:
            Dictionary containing EMG signals and metadata
        """
        if gesture_id not in self.gestures:
            raise ValueError(f"Invalid gesture ID: {gesture_id}")
        
        # Get gesture-specific parameters
        gesture_params = self.gesture_activations[gesture_id]
        base_activation = gesture_params['activation']
        
        # Adjust activation based on force level
        effective_activation = base_activation * force_level
        
        # Generate biomechanical signals
        spike_trains, muscle_force = self.biomechanical.simulate_muscle_dynamics(
            activation=effective_activation,
            duration=duration,
            fatigue_factor=fatigue_factor
        )
        
        # Generate electrical signals
        emg_signals = self.electrical.generate_emg_signals(spike_trains)
        
        # Create metadata
        metadata = {
            'gesture_id': gesture_id,
            'gesture_name': self.gestures[gesture_id],
            'force_level': force_level,
            'muscle_force': muscle_force,
            'fatigue_factor': fatigue_factor,
            'tissue_depth': tissue_depth if tissue_depth is not None 
                           else np.mean(self.electrical.muscle_depth),
            'sampling_rate': self.sampling_rate,
            'duration': duration
        }
        
        return {
            'emg_signals': emg_signals,  # shape: (num_channels, num_samples)
            'metadata': metadata
        }
    
    def generate_dataset(self, num_samples: int, output_format: str = 'hdf5',
                        output_path: Optional[str] = None, 
                        gesture_distribution: Optional[List[float]] = None,
                        force_range: Tuple[float, float] = (0.3, 1.0),
                        duration_range: Tuple[float, float] = (0.5, 2.0),
                        enable_progress: bool = True) -> Union[str, Dict]:
        """
        Generate a complete synthetic EMG dataset.
        
        Args:
            num_samples: Number of samples to generate
            output_format: 'hdf5', 'csv', or 'dict'
            output_path: Output file path (required for file formats)
            gesture_distribution: Probability distribution for gestures
            force_range: Range of force levels
            duration_range: Range of signal durations
            enable_progress: Show progress bar
            
        Returns:
            File path or dictionary containing the dataset
        """
        if output_format in ['hdf5', 'csv'] and output_path is None:
            raise ValueError("output_path required for file formats")
        
        # Set gesture distribution
        if gesture_distribution is None:
            gesture_distribution = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]
        
        if len(gesture_distribution) != len(self.gestures):
            raise ValueError("gesture_distribution length must match number of gestures")
        
        gesture_distribution = np.array(gesture_distribution)
        gesture_distribution /= gesture_distribution.sum()
        
        # Initialize storage
        all_emg_signals = []
        all_metadata = []
        
        # Generate samples
        iterator = tqdm(range(num_samples), desc="Generating samples") if enable_progress else range(num_samples)
        
        start_time = time.time()
        
        for i in iterator:
            # Sample gesture
            gesture_id = np.random.choice(len(self.gestures), p=gesture_distribution)
            
            # Sample parameters
            force_level = np.random.uniform(*force_range)
            duration = np.random.uniform(*duration_range)
            fatigue_factor = np.random.uniform(0.7, 1.0)
            tissue_depth = np.random.uniform(10, 25)  # mm
            
            # Generate sample
            sample = self.generate_single_sample(
                gesture_id=gesture_id,
                force_level=force_level,
                duration=duration,
                fatigue_factor=fatigue_factor,
                tissue_depth=tissue_depth
            )
            
            all_emg_signals.append(sample['emg_signals'])
            all_metadata.append(sample['metadata'])
        
        generation_time = time.time() - start_time
        
        # Store dataset
        if output_format == 'hdf5':
            return self._save_hdf5(all_emg_signals, all_metadata, output_path, generation_time)
        elif output_format == 'csv':
            return self._save_csv(all_emg_signals, all_metadata, output_path, generation_time)
        elif output_format == 'dict':
            return {
                'emg_signals': all_emg_signals,
                'metadata': all_metadata,
                'generation_time': generation_time
            }
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _save_hdf5(self, emg_signals: List[np.ndarray], metadata: List[Dict],
                  output_path: str, generation_time: float) -> str:
        """Save dataset to HDF5 format."""
        with h5py.File(output_path, 'w') as f:
            # Store EMG signals
            emg_group = f.create_group('emg_signals')
            for i, signals in enumerate(emg_signals):
                emg_group.create_dataset(f'sample_{i}', data=signals)
            
            # Store metadata
            meta_group = f.create_group('metadata')
            
            # Convert metadata lists to arrays
            gesture_ids = [m['gesture_id'] for m in metadata]
            gesture_names = [m['gesture_name'] for m in metadata]
            force_levels = [m['force_level'] for m in metadata]
            fatigue_factors = [m['fatigue_factor'] for m in metadata]
            tissue_depths = [m['tissue_depth'] for m in metadata]
            durations = [m['duration'] for m in metadata]
            
            meta_group.create_dataset('gesture_ids', data=gesture_ids)
            meta_group.create_dataset('gesture_names', data=gesture_names)
            meta_group.create_dataset('force_levels', data=force_levels)
            meta_group.create_dataset('fatigue_factors', data=fatigue_factors)
            meta_group.create_dataset('tissue_depths', data=tissue_depths)
            meta_group.create_dataset('durations', data=durations)
            
            # Store muscle forces (variable length, need special handling)
            force_group = meta_group.create_group('muscle_forces')
            for i, force in enumerate([m['muscle_force'] for m in metadata]):
                force_group.create_dataset(f'sample_{i}', data=force)
            
            # Store generation info
            info_group = f.create_group('info')
            info_group.attrs['num_samples'] = len(emg_signals)
            info_group.attrs['num_channels'] = self.num_channels
            info_group.attrs['sampling_rate'] = self.sampling_rate
            info_group.attrs['num_motor_units'] = self.num_motor_units
            info_group.attrs['generation_time'] = generation_time
        
        return output_path
    
    def _save_csv(self, emg_signals: List[np.ndarray], metadata: List[Dict],
                 output_path: str, generation_time: float) -> str:
        """Save dataset to CSV format (simplified)."""
        # This is a simplified CSV export - for large datasets, HDF5 is recommended
        rows = []
        
        for i, (signals, meta) in enumerate(zip(emg_signals, metadata)):
            # Take mean signal across channels for simplicity
            mean_signal = np.mean(signals, axis=0)
            
            # Create multiple rows per sample (one per time point)
            for t in range(len(mean_signal)):
                row = {
                    'sample_id': i,
                    'time_point': t / self.sampling_rate,
                    'emg_value': mean_signal[t],
                    'gesture_id': meta['gesture_id'],
                    'gesture_name': meta['gesture_name'],
                    'force_level': meta['force_level'],
                    'fatigue_factor': meta['fatigue_factor'],
                    'tissue_depth': meta['tissue_depth'],
                    'duration': meta['duration']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def benchmark_performance(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark generation performance.
        
        Args:
            num_samples: Number of samples for benchmark
            
        Returns:
            Performance metrics
        """
        print(f"Benchmarking BioSynth-EMG with {num_samples} samples...")
        
        start_time = time.time()
        dataset = self.generate_dataset(
            num_samples=num_samples,
            output_format='dict',
            enable_progress=False
        )
        total_time = time.time() - start_time
        
        # Calculate metrics
        samples_per_second = num_samples / total_time
        avg_time_per_sample = total_time / num_samples
        
        # Calculate signal duration per sample
        total_signal_duration = sum(meta['duration'] for meta in dataset['metadata'])
        signal_time_ratio = total_signal_duration / total_time
        
        metrics = {
            'total_time': total_time,
            'samples_per_second': samples_per_second,
            'avg_time_per_sample': avg_time_per_sample,
            'signal_time_ratio': signal_time_ratio,
            'target_performance': avg_time_per_sample < 0.001  # < 1ms target
        }
        
        print(f"Performance Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Samples per second: {samples_per_second:.1f}")
        print(f"  Time per sample: {avg_time_per_sample*1000:.2f}ms")
        print(f"  Signal time ratio: {signal_time_ratio:.1f}x")
        print(f"  Target met (<1ms): {metrics['target_performance']}")
        
        return metrics

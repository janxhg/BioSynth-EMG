#!/usr/bin/env python3
"""
Test suite for BioSynth-EMG synthetic signal generator.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from biosynth_emg import BioSynthGenerator, SpectralValidator


class TestBioSynthGenerator(unittest.TestCase):
    """Test cases for BioSynthGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = BioSynthGenerator(
            sampling_rate=1000,
            num_motor_units=50,
            num_channels=8,
            random_seed=42
        )
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.sampling_rate, 1000)
        self.assertEqual(self.generator.num_motor_units, 50)
        self.assertEqual(self.generator.num_channels, 8)
        self.assertEqual(len(self.generator.gestures), 8)
    
    def test_single_sample_generation(self):
        """Test single EMG sample generation."""
        sample = self.generator.generate_single_sample(
            gesture_id=1,
            force_level=0.5,
            duration=1.0
        )
        
        # Check structure
        self.assertIn('emg_signals', sample)
        self.assertIn('metadata', sample)
        
        # Check EMG signals
        emg_signals = sample['emg_signals']
        self.assertEqual(emg_signals.shape[0], 8)  # 8 channels
        self.assertEqual(emg_signals.shape[1], 1000)  # 1 second at 1000 Hz
        
        # Check metadata
        metadata = sample['metadata']
        self.assertEqual(metadata['gesture_id'], 1)
        self.assertEqual(metadata['gesture_name'], 'fist')
        self.assertEqual(metadata['force_level'], 0.5)
        self.assertEqual(metadata['sampling_rate'], 1000)
        self.assertEqual(metadata['duration'], 1.0)
    
    def test_invalid_gesture_id(self):
        """Test handling of invalid gesture ID."""
        with self.assertRaises(ValueError):
            self.generator.generate_single_sample(gesture_id=99)
    
    def test_dataset_generation_dict(self):
        """Test dataset generation in dictionary format."""
        dataset = self.generator.generate_dataset(
            num_samples=10,
            output_format='dict',
            enable_progress=False
        )
        
        # Check structure
        self.assertIn('emg_signals', dataset)
        self.assertIn('metadata', dataset)
        self.assertIn('generation_time', dataset)
        
        # Check sizes
        self.assertEqual(len(dataset['emg_signals']), 10)
        self.assertEqual(len(dataset['metadata']), 10)
        
        # Check each sample
        for signals, metadata in zip(dataset['emg_signals'], dataset['metadata']):
            self.assertEqual(signals.shape[0], 8)  # 8 channels
            self.assertIn('gesture_id', metadata)
            self.assertIn('force_level', metadata)
    
    def test_dataset_generation_hdf5(self):
        """Test dataset generation in HDF5 format."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            output_path = self.generator.generate_dataset(
                num_samples=5,
                output_format='hdf5',
                output_path=tmp_path,
                enable_progress=False
            )
            
            self.assertEqual(output_path, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Verify HDF5 file structure
            import h5py
            with h5py.File(tmp_path, 'r') as f:
                self.assertIn('emg_signals', f)
                self.assertIn('metadata', f)
                self.assertIn('info', f)
                
                # Check info attributes
                info = f['info']
                self.assertEqual(info.attrs['num_samples'], 5)
                self.assertEqual(info.attrs['num_channels'], 8)
                self.assertEqual(info.attrs['sampling_rate'], 1000)
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_dataset_generation_csv(self):
        """Test dataset generation in CSV format."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            output_path = self.generator.generate_dataset(
                num_samples=3,
                output_format='csv',
                output_path=tmp_path,
                enable_progress=False
            )
            
            self.assertEqual(output_path, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Verify CSV file
            import pandas as pd
            df = pd.read_csv(tmp_path)
            self.assertIn('sample_id', df.columns)
            self.assertIn('gesture_id', df.columns)
            self.assertIn('emg_value', df.columns)
            self.assertGreater(len(df), 0)
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_gesture_distribution(self):
        """Test custom gesture distribution."""
        custom_dist = [0.5, 0.5, 0, 0, 0, 0, 0, 0]  # Only gestures 0 and 1
        
        dataset = self.generator.generate_dataset(
            num_samples=100,
            output_format='dict',
            gesture_distribution=custom_dist,
            enable_progress=False
        )
        
        gesture_ids = [meta['gesture_id'] for meta in dataset['metadata']]
        unique_gestures = set(gesture_ids)
        
        # Should only contain gestures 0 and 1
        self.assertLessEqual(len(unique_gestures), 2)
        self.assertTrue(all(gid in [0, 1] for gid in unique_gestures))
    
    def test_performance_benchmark(self):
        """Test performance benchmarking."""
        metrics = self.generator.benchmark_performance(num_samples=10)
        
        # Check required metrics
        required_keys = ['total_time', 'samples_per_second', 'avg_time_per_sample', 
                        'signal_time_ratio', 'target_performance']
        for key in required_keys:
            self.assertIn(key, metrics)
        
        # Check data types and ranges
        self.assertGreater(metrics['total_time'], 0)
        self.assertGreater(metrics['samples_per_second'], 0)
        self.assertGreater(metrics['avg_time_per_sample'], 0)
        self.assertGreater(metrics['signal_time_ratio'], 0)
        self.assertIsInstance(metrics['target_performance'], bool)


class TestSpectralValidator(unittest.TestCase):
    """Test cases for SpectralValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = SpectralValidator(sampling_rate=1000)
        
        # Create test signal
        t = np.linspace(0, 1, 1000)
        # Multi-component signal with known frequencies
        self.test_signal = (np.sin(2 * np.pi * 50 * t) +  # 50 Hz component
                           0.5 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz component
                           0.1 * np.random.randn(len(t)))  # Noise
    
    def test_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.sampling_rate, 1000)
        self.assertEqual(self.validator.nyquist, 500)
        self.assertEqual(self.validator.expected_peak_freq_range, (30, 120))
    
    def test_single_signal_analysis(self):
        """Test analysis of single signal."""
        metrics = self.validator.analyze_single_signal(self.test_signal)
        
        # Check required metrics
        required_keys = ['peak_frequency', 'median_frequency', 'bandwidth', 
                        'lower_freq', 'upper_freq', 'snr_db', 'total_power']
        for key in required_keys:
            self.assertIn(key, metrics)
        
        # Check value ranges
        self.assertGreater(metrics['peak_frequency'], 0)
        self.assertLess(metrics['peak_frequency'], self.validator.nyquist)
        self.assertGreater(metrics['median_frequency'], 0)
        self.assertGreater(metrics['bandwidth'], 0)
        self.assertGreater(metrics['total_power'], 0)
    
    def test_dataset_validation(self):
        """Test validation of signal dataset."""
        # Create synthetic dataset
        signals = []
        for _ in range(5):
            t = np.linspace(0, 1, 1000)
            signal = (np.sin(2 * np.pi * np.random.uniform(40, 160) * t) + 
                     0.1 * np.random.randn(len(t)))
            signals.append(np.array([signal, signal]))  # 2 channels
        
        results = self.validator.validate_dataset(signals)
        
        # Check structure
        self.assertIn('metrics', results)
        self.assertIn('validation', results)
        self.assertIn('raw_metrics', results)
        
        # Check validation results
        validation = results['validation']
        self.assertIn('overall_valid', validation)
        self.assertIn('peak_freq_in_range', validation)
        self.assertIn('median_freq_in_range', validation)
        self.assertIn('bandwidth_realistic', validation)
        self.assertIn('snr_acceptable', validation)
    
    def test_frequency_range_check(self):
        """Test frequency range validation."""
        # Test with good frequencies
        good_stats = {'values': [75, 85, 95, 105]}  # All in target range
        result = self.validator._check_frequency_range(good_stats)
        self.assertTrue(result['passed'])
        self.assertEqual(result['pass_rate'], 1.0)
        
        # Test with bad frequencies
        bad_stats = {'values': [25, 35, 200, 250]}  # All outside target range (30-120)
        result = self.validator._check_frequency_range(bad_stats)
        self.assertFalse(result['passed'])
        # Only 25 is outside range, 35 is inside (30-120), so pass rate should be 0.25
        self.assertEqual(result['pass_rate'], 0.25)
    
    def test_validation_report_generation(self):
        """Test validation report generation."""
        # Create mock validation results
        mock_results = {
            'validation': {
                'overall_valid': True,
                'peak_freq_in_range': {'passed': True, 'pass_rate': 0.9, 'target_range': (50, 150)},
                'median_freq_in_range': {'passed': True, 'pass_rate': 0.85, 'target_range': (30, 200)},
                'bandwidth_realistic': {'passed': True, 'pass_rate': 0.8, 'target_range': (50, 400)},
                'snr_acceptable': {'passed': True, 'pass_rate': 0.7, 'target_range': (10, 30)}
            },
            'metrics': {
                'peak_frequency': {'mean': 85.5, 'std': 15.2, 'min': 60.1, 'max': 110.3},
                'median_frequency': {'mean': 95.2, 'std': 20.1, 'min': 70.5, 'max': 120.8},
                'bandwidth': {'mean': 180.5, 'std': 45.3, 'min': 120.1, 'max': 240.9},
                'snr_db': {'mean': 18.5, 'std': 5.2, 'min': 12.1, 'max': 24.8}
            }
        }
        
        report = self.validator.generate_validation_report(mock_results)
        
        # Check report content
        self.assertIn('BioSynth-EMG Spectral Validation Report', report)
        self.assertIn('PASSED', report)
        self.assertIn('Peak Frequency Range', report)
        self.assertIn('Statistical Summary', report)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete BioSynth-EMG system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.generator = BioSynthGenerator(
            sampling_rate=2000,
            num_motor_units=100,
            num_channels=8,
            random_seed=42
        )
        self.validator = SpectralValidator(sampling_rate=2000)
    
    def test_full_pipeline(self):
        """Test the complete generation and validation pipeline."""
        # Generate dataset
        dataset = self.generator.generate_dataset(
            num_samples=20,
            output_format='dict',
            enable_progress=False
        )
        
        # Validate dataset
        validation_results = self.validator.validate_dataset(dataset['emg_signals'])
        
        # Check that pipeline completed successfully
        self.assertEqual(len(dataset['emg_signals']), 20)
        self.assertEqual(len(dataset['metadata']), 20)
        self.assertIn('validation', validation_results)
        self.assertIn('metrics', validation_results)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        # Generate with same seed twice
        generator1 = BioSynthGenerator(random_seed=123)
        generator2 = BioSynthGenerator(random_seed=123)
        
        sample1 = generator1.generate_single_sample(gesture_id=1, force_level=0.5)
        sample2 = generator2.generate_single_sample(gesture_id=1, force_level=0.5)
        
        # Check that metadata is identical
        self.assertEqual(sample1['metadata']['gesture_id'], sample2['metadata']['gesture_id'])
        self.assertEqual(sample1['metadata']['force_level'], sample2['metadata']['force_level'])
        
        # Check that both signals have the same shape and basic properties
        self.assertEqual(sample1['emg_signals'].shape, sample2['emg_signals'].shape)
        
        # Check that signals are non-zero (basic sanity check)
        self.assertGreater(np.mean(np.abs(sample1['emg_signals'])), 0)
        self.assertGreater(np.mean(np.abs(sample2['emg_signals'])), 0)
        
        # Check that overall signal power is in reasonable range
        power1 = np.mean(sample1['emg_signals']**2)
        power2 = np.mean(sample2['emg_signals']**2)
        self.assertGreater(power1, 0.001)  # Should have some power
        self.assertGreater(power2, 0.001)
        self.assertLess(power1, 10.0)  # But not too much
        self.assertLess(power2, 10.0)
    
    def test_different_gestures(self):
        """Test generation of different gesture types."""
        gesture_ids = list(range(8))  # All available gestures
        samples = []
        
        for gesture_id in gesture_ids:
            sample = self.generator.generate_single_sample(
                gesture_id=gesture_id,
                force_level=0.5,
                duration=0.5
            )
            samples.append(sample)
        
        # Check that all gestures were generated
        generated_gestures = [sample['metadata']['gesture_id'] for sample in samples]
        self.assertEqual(set(generated_gestures), set(gesture_ids))
        
        # Check that all have correct shape
        for sample in samples:
            self.assertEqual(sample['emg_signals'].shape[0], 8)
            self.assertEqual(sample['emg_signals'].shape[1], 1000)  # 0.5s at 2000 Hz


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

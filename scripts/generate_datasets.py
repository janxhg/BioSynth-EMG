#!/usr/bin/env python3
"""
BioSynth-EMG Dataset Generator Script
Generates synthetic EMG datasets of different sizes for research and training.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from biosynth_emg import BioSynthGenerator, SpectralValidator


def generate_dataset(size_name: str, num_samples: int, output_dir: str = "datasets", 
                    seed: int = 42, validate: bool = True):
    """
    Generate a synthetic EMG dataset of specified size.
    
    Args:
        size_name: Name for the dataset size (e.g., '1k', '10k')
        num_samples: Number of samples to generate
        output_dir: Directory to save the dataset
        seed: Random seed for reproducibility
        validate: Whether to perform spectral validation
    """
    print(f"\n{'='*60}")
    print(f"Generating BioSynth-EMG Dataset: {size_name.upper()}")
    print(f"{'='*60}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize generator
    print(f"\n1. Initializing BioSynth-EMG generator...")
    generator = BioSynthGenerator(
        sampling_rate=2000,
        num_motor_units=100,
        num_channels=8,
        random_seed=seed
    )
    
    # Generate dataset
    print(f"\n2. Generating {num_samples:,} samples...")
    start_time = time.time()
    
    dataset = generator.generate_dataset(
        num_samples=num_samples,
        output_format='dict',
        gesture_distribution=None,  # Uniform distribution
        force_range=(0.3, 1.0),
        duration_range=(0.5, 2.0),
        enable_progress=True
    )
    
    generation_time = time.time() - start_time
    print(f"   Dataset generated in {generation_time:.2f} seconds")
    print(f"   Average time per sample: {generation_time/num_samples*1000:.2f} ms")
    
    # Save dataset
    print(f"\n3. Saving dataset...")
    dataset_filename = f"biosynth_emg_{size_name}_dataset.h5"
    dataset_path = output_path / dataset_filename
    
    with h5py.File(dataset_path, 'w') as f:
        # Save EMG signals (handle variable lengths)
        emg_dset = f.create_dataset('emg_signals', 
                                   shape=(num_samples, 8, 2000),  # Fixed shape
                                   dtype=np.float32,
                                   compression='gzip')
        
        # Pad or truncate signals to fixed length
        for i in range(num_samples):
            signal = dataset['emg_signals'][i]
            if signal.shape[1] < 2000:
                # Pad with zeros
                padded = np.zeros((8, 2000), dtype=np.float32)
                padded[:, :signal.shape[1]] = signal
                emg_dset[i] = padded
            else:
                # Truncate
                emg_dset[i] = signal[:, :2000].astype(np.float32)
        
        # Extract labels and forces from metadata
        gesture_labels = np.array([meta['gesture_id'] for meta in dataset['metadata']])
        force_levels = np.array([meta['force_level'] for meta in dataset['metadata']])
        
        # Save labels and forces
        f.create_dataset('gesture_labels', data=gesture_labels)
        f.create_dataset('force_levels', data=force_levels)
        
        # Save metadata
        metadata_group = f.create_group('metadata')
        for i, meta in enumerate(dataset['metadata']):
            meta_group = metadata_group.create_group(f'sample_{i}')
            for key, value in meta.items():
                if isinstance(value, (int, float, str)):
                    meta_group.attrs[key] = value
                else:
                    meta_group.create_dataset(key, data=value)
        
        # Save dataset info
        info_group = f.create_group('dataset_info')
        info_group.attrs['num_samples'] = num_samples
        info_group.attrs['num_channels'] = 8
        info_group.attrs['sampling_rate'] = 2000
        info_group.attrs['generation_time'] = generation_time
        info_group.attrs['created_at'] = datetime.now().isoformat()
        info_group.attrs['random_seed'] = seed
    
    print(f"   Dataset saved to: {dataset_path}")
    print(f"   File size: {dataset_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Spectral validation (optional)
    if validate:
        print(f"\n4. Performing spectral validation...")
        validator = SpectralValidator(sampling_rate=2000)
        start_time = time.time()
        
        # Sample subset for validation (to save time)
        validation_samples = min(100, num_samples)
        sample_indices = np.random.choice(num_samples, validation_samples, replace=False)
        
        # Convert to numpy array with proper shape
        validation_signals = []
        for idx in sample_indices:
            signal = dataset['emg_signals'][idx]
            # Ensure consistent shape
            if signal.shape[1] < 2000:
                padded = np.zeros((8, 2000))
                padded[:, :signal.shape[1]] = signal
                validation_signals.append(padded)
            else:
                validation_signals.append(signal[:, :2000])
        
        validation_signals = np.array(validation_signals)
        
        validation_results = validator.validate_dataset(validation_signals)
        validation_time = time.time() - start_time
        
        print(f"   Validation completed in {validation_time:.2f} seconds")
        print(f"   Overall validation: {'PASSED' if validation_results['validation']['overall_valid'] else 'FAILED'}")
        
        # Save validation report
        report_filename = f"biosynth_emg_{size_name}_validation.txt"
        report_path = output_path / report_filename
        
        with open(report_path, 'w') as f:
            f.write(f"BioSynth-EMG Dataset Validation Report\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Dataset: {size_name.upper()} ({num_samples:,} samples)\n")
            f.write(f"Validation samples: {validation_samples}\n")
            f.write(f"Validation time: {validation_time:.2f} seconds\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
            
            f.write(f"Overall Validation: {'PASSED' if validation_results['validation']['overall_valid'] else 'FAILED'}\n\n")
            
            for check_name, result in validation_results['validation'].items():
                if check_name != 'overall_valid':
                    f.write(f"{check_name.replace('_', ' ').title()}: {'PASSED' if result['passed'] else 'FAILED'}\n")
                    f.write(f"  Pass Rate: {result['pass_rate']:.1%}\n")
                    if 'target_range' in result:
                        f.write(f"  Target Range: {result['target_range']}\n")
                    f.write("\n")
            
            f.write(f"Spectral Metrics Summary:\n")
            f.write(f"{'-'*30}\n")
            for metric_name, stats in validation_results['metrics'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    f.write(f"{metric_name.replace('_', ' ').title()}: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
        
        print(f"   Validation report saved to: {report_path}")
    
    # Generate CSV summary
    print(f"\n5. Generating CSV summary...")
    csv_filename = f"biosynth_emg_{size_name}_summary.csv"
    csv_path = output_path / csv_filename
    
    summary_data = []
    for i in range(min(1000, num_samples)):  # First 1000 samples for summary
        summary_data.append({
            'sample_id': i,
            'gesture_id': dataset['metadata'][i]['gesture_id'],
            'gesture_name': generator.gestures[dataset['metadata'][i]['gesture_id']],
            'force_level': dataset['metadata'][i]['force_level'],
            'signal_rms': np.sqrt(np.mean(dataset['emg_signals'][i]**2)),
            'signal_mean': np.mean(dataset['emg_signals'][i]),
            'signal_std': np.std(dataset['emg_signals'][i])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_path, index=False)
    
    print(f"   CSV summary saved to: {csv_path}")
    
    print(f"\n{'='*60}")
    print(f"Dataset {size_name.upper()} generation completed successfully!")
    print(f"{'='*60}")
    
    return {
        'dataset_path': str(dataset_path),
        'validation_path': str(report_path) if validate else None,
        'summary_path': str(csv_path),
        'generation_time': generation_time,
        'file_size_mb': dataset_path.stat().st_size / 1024 / 1024
    }


def main():
    """Main function to generate datasets."""
    parser = argparse.ArgumentParser(description='Generate BioSynth-EMG datasets')
    parser.add_argument('--sizes', nargs='+', default=['1k', '10k'], 
                       help='Dataset sizes to generate (e.g., 1k 10k 100k)')
    parser.add_argument('--output-dir', default='datasets', 
                       help='Output directory for datasets')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--no-validation', action='store_true', 
                       help='Skip spectral validation')
    
    args = parser.parse_args()
    
    # Size mapping
    size_mapping = {
        '1k': 1000,
        '10k': 10000,
        '100k': 100000,
        '1m': 1000000
    }
    
    print("BioSynth-EMG Dataset Generator")
    print("="*40)
    print(f"Sizes to generate: {', '.join(args.sizes)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Spectral validation: {'Disabled' if args.no_validation else 'Enabled'}")
    
    results = []
    total_start_time = time.time()
    
    for size_name in args.sizes:
        if size_name not in size_mapping:
            print(f"Warning: Unknown size '{size_name}'. Skipping.")
            continue
        
        num_samples = size_mapping[size_name]
        
        try:
            result = generate_dataset(
                size_name=size_name,
                num_samples=num_samples,
                output_dir=args.output_dir,
                seed=args.seed,
                validate=not args.no_validation
            )
            results.append(result)
        except Exception as e:
            print(f"Error generating {size_name} dataset: {e}")
            continue
    
    # Final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Datasets generated: {len(results)}")
    
    for result in results:
        print(f"  - {result['dataset_path']}")
        print(f"    Size: {result['file_size_mb']:.1f} MB")
        print(f"    Time: {result['generation_time']:.2f} seconds")
    
    print(f"\nAll datasets saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

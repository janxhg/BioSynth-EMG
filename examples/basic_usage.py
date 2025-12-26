#!/usr/bin/env python3
"""
Basic usage example for BioSynth-EMG synthetic signal generator.
Demonstrates core functionality and performance benchmarking.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Add the parent directory to the path to import biosynth_emg
import sys
sys.path.append(str(Path(__file__).parent.parent))

from biosynth_emg import BioSynthGenerator, SpectralValidator


def main():
    """Main demonstration function."""
    print("BioSynth-EMG Basic Usage Example")
    print("=" * 40)
    
    # Initialize generator
    print("1. Initializing BioSynth-EMG generator...")
    generator = BioSynthGenerator(
        sampling_rate=2000,
        num_motor_units=100,
        num_channels=8,
        random_seed=42
    )
    
    # Generate single sample
    print("\n2. Generating single EMG sample...")
    sample = generator.generate_single_sample(
        gesture_id=1,  # Fist gesture
        force_level=0.7,
        duration=1.0,
        fatigue_factor=0.9
    )
    
    print(f"   Generated EMG signals shape: {sample['emg_signals'].shape}")
    print(f"   Gesture: {sample['metadata']['gesture_name']}")
    print(f"   Force level: {sample['metadata']['force_level']}")
    print(f"   Sampling rate: {sample['metadata']['sampling_rate']} Hz")
    
    # Generate small dataset
    print("\n3. Generating small dataset (50 samples)...")
    start_time = time.time()
    
    dataset = generator.generate_dataset(
        num_samples=50,
        output_format='dict',
        enable_progress=True
    )
    
    generation_time = time.time() - start_time
    print(f"   Dataset generated in {generation_time:.3f} seconds")
    print(f"   Average time per sample: {generation_time/50*1000:.2f} ms")
    
    # Spectral validation
    print("\n4. Performing spectral validation...")
    validator = SpectralValidator(sampling_rate=2000)
    
    # Validate first few samples
    validation_results = validator.validate_dataset(dataset['emg_signals'][:10])
    
    print(f"   Overall validation: {'PASSED' if validation_results['validation']['overall_valid'] else 'FAILED'}")
    
    # Print key metrics
    metrics = validation_results['metrics']
    print(f"   Peak frequency: {metrics['peak_frequency']['mean']:.1f} ± {metrics['peak_frequency']['std']:.1f} Hz")
    print(f"   Median frequency: {metrics['median_frequency']['mean']:.1f} ± {metrics['median_frequency']['std']:.1f} Hz")
    print(f"   Bandwidth: {metrics['bandwidth']['mean']:.1f} ± {metrics['bandwidth']['std']:.1f} Hz")
    print(f"   SNR: {metrics['snr_db']['mean']:.1f} ± {metrics['snr_db']['std']:.1f} dB")
    
    # Performance benchmark
    print("\n5. Performance benchmark...")
    benchmark_results = generator.benchmark_performance(num_samples=100)
    
    print(f"   Target performance (<1ms per sample): {'MET' if benchmark_results['target_performance'] else 'NOT MET'}")
    print(f"   Signal generation speed: {benchmark_results['signal_time_ratio']:.1f}x real-time")
    
    # Visualize sample signals
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Multi-channel EMG signals
    ax1 = axes[0, 0]
    emg_signals = sample['emg_signals']
    time_axis = np.arange(emg_signals.shape[1]) / generator.sampling_rate
    
    for ch in range(min(4, emg_signals.shape[0])):  # Show first 4 channels
        ax1.plot(time_axis[:1000], emg_signals[ch, :1000] + ch*0.5, 
                label=f'Channel {ch+1}', alpha=0.8)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('EMG Amplitude (offset)')
    ax1.set_title('Multi-channel EMG Signals')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Frequency spectrum
    ax2 = axes[0, 1]
    from scipy.fft import fft, fftfreq
    
    # Analyze first channel
    signal_ch1 = emg_signals[0]
    fft_vals = fft(signal_ch1 - np.mean(signal_ch1))
    fft_freq = fftfreq(len(signal_ch1), 1/generator.sampling_rate)
    
    positive_freq_idx = (fft_freq > 0) & (fft_freq < 500)
    ax2.plot(fft_freq[positive_freq_idx], np.abs(fft_vals[positive_freq_idx]))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Spectrum (Channel 1)')
    ax2.grid(True)
    ax2.axvspan(50, 150, alpha=0.2, color='green', label='Target range')
    ax2.legend()
    
    # Plot 3: Muscle force trajectory
    ax3 = axes[1, 0]
    muscle_force = sample['metadata']['muscle_force']
    force_time = np.arange(len(muscle_force)) / generator.sampling_rate
    
    ax3.plot(force_time, muscle_force, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Normalized Force')
    ax3.set_title('Simulated Muscle Force')
    ax3.grid(True)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Dataset statistics
    ax4 = axes[1, 1]
    
    # Collect gesture distribution
    gesture_ids = [meta['gesture_id'] for meta in dataset['metadata']]
    gesture_names = [meta['gesture_name'] for meta in dataset['metadata']]
    force_levels = [meta['force_level'] for meta in dataset['metadata']]
    
    # Gesture distribution
    unique_gestures, counts = np.unique(gesture_ids, return_counts=True)
    gesture_labels = [dataset['metadata'][gesture_ids.index(g)]['gesture_name'] for g in unique_gestures]
    
    ax4.bar(range(len(unique_gestures)), counts)
    ax4.set_xlabel('Gesture')
    ax4.set_ylabel('Count')
    ax4.set_title('Gesture Distribution in Dataset')
    ax4.set_xticks(range(len(unique_gestures)))
    ax4.set_xticklabels(gesture_labels, rotation=45)
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    fig_path = output_dir / "biosynth_demo.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved to: {fig_path}")
    
    # Save validation report
    report_path = output_dir / "validation_report.txt"
    validator.generate_validation_report(validation_results, report_path)
    print(f"   Validation report saved to: {report_path}")
    
    plt.show()
    
    print("\n" + "=" * 40)
    print("BioSynth-EMG demonstration completed successfully!")
    print(f"Check the '{output_dir}' directory for outputs.")


if __name__ == "__main__":
    main()

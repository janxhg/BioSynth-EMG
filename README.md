# BioSynth-EMG: Physics-Informed Generative Framework for Synthetic Myoelectric Signals

A physics-based synthetic EMG signal generator for training low-latency neural networks in prosthetic control systems.

## Features

- **Biomechanical Layer**: Motor unit firing rate simulation using Poisson distribution
- **Electrical Propagation Layer**: MUAP generation with Hermite functions
- **Realistic Noise**: Gaussian noise, power line interference, and movement artifacts
- **High Performance**: Optimized for GTX 1650 (<1ms generation for 1s signal)
- **Dataset Generation**: 8-channel EMG with gesture labels and force values

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install biosynth-emg
```

### Option 2: Install from Source

```bash
git clone https://github.com/janxhg/BioSynth-EMG.git
cd BioSynth-EMG
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from biosynth_emg import BioSynthGenerator

generator = BioSynthGenerator()
data = generator.generate_dataset(num_samples=1000, duration=1.0)
```

## Paper

Based on "BioSynth-EMG: A Physics-Informed Generative Framework for Synthetic Myoelectric Signal Synthesis and Prothetic Control Benchmarking"

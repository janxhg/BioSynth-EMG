# BioSynth-EMG Dataset Generation Scripts

This directory contains scripts for generating synthetic EMG datasets of various sizes.

## generate_datasets.py

A comprehensive script for generating BioSynth-EMG datasets with spectral validation and reporting.

### Usage

```bash
# Generate default datasets (1k and 10k samples)
python scripts/generate_datasets.py

# Generate specific sizes
python scripts/generate_datasets.py --sizes 1k 10k 100k

# Custom output directory
python scripts/generate_datasets.py --output-dir my_datasets

# Skip validation for faster generation
python scripts/generate_datasets.py --no-validation

# Custom random seed
python scripts/generate_datasets.py --seed 123
```

### Output Files

For each dataset size, the script generates:

1. **HDF5 Dataset**: `biosynth_emg_{size}_dataset.h5`
   - EMG signals (8 channels)
   - Gesture labels and force levels
   - Complete metadata
   - Dataset information

2. **Validation Report**: `biosynth_emg_{size}_validation.txt`
   - Spectral validation results
   - Pass/fail status for each metric
   - Statistical summaries

3. **CSV Summary**: `biosynth_emg_{size}_summary.csv`
   - First 1000 samples summary
   - Basic signal statistics
   - Gesture and force information

### Supported Sizes

- `1k`: 1,000 samples
- `10k`: 10,000 samples  
- `100k`: 100,000 samples
- `1m`: 1,000,000 samples

### Examples

```bash
# Generate small dataset for testing
python scripts/generate_datasets.py --sizes 1k

# Generate large research dataset
python scripts/generate_datasets.py --sizes 100k --output-dir research_data

# Generate multiple sizes with custom seed
python scripts/generate_datasets.py --sizes 1k 10k 100k --seed 42
```

### Performance

Typical generation times on GTX 1650:
- 1k samples: ~30 seconds
- 10k samples: ~5 minutes
- 100k samples: ~50 minutes

File sizes (approximate):
- 1k dataset: ~50 MB
- 10k dataset: ~500 MB
- 100k dataset: ~5 GB

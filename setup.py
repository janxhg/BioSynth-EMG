from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("LICENSE", "r", encoding="utf-8") as fh:
    license_text = fh.read()

setup(
    name="biosynth-emg",
    version="0.1.0",
    description="Physics-informed synthetic EMG signal generator for prosthetic control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NetechAI",
    author_email="joaquinsturzt26@gmail.com",
    url="https://github.com/janxhg/BioSynth-EMG",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "h5py>=3.1.0",
        "matplotlib>=3.4.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="emg, electromyography, synthetic-data, prosthetics, machine-learning, biomedical",
    project_urls={
        "Bug Reports": "https://github.com/janxhg/BioSynth-EMG/issues",
        "Source": "https://github.com/janxhg/BioSynth-EMG",
        "Paper": "https://github.com/janxhg/BioSynth-EMG/blob/main/paper.md",
    },
)

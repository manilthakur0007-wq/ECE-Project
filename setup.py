from setuptools import setup, find_packages

setup(
    name="ecg_arrhythmia_detector",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "wfdb>=4.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
    ],
)

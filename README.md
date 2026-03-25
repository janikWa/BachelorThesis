# Bachelor Thesis: Heavy-Tailed Distributions in Neural Networks

This repository contains the complete codebase for my Bachelor's thesis. The project focuses on investigating weight distributions in neural networks, specifically analyzing heavy-tailed $\alpha$-stable distributions enforced by explicit regularization and their impact on the generalization capabilities of the networks and their overall performance. 
## Main Concepts & Features

- **Custom Regularizers:** In addition to standard methods (L1/Lasso, L2/Ridge), this project implements specialized regularization techniques targeting $\alpha$-stable distributions (e.g., **Hill regularization** and parabolic approaches), which directly influence the tail index of the weights.
- **Grokking:** A dedicated module isolates and demonstrates the *grokking* phenomenon (delayed generalization after extended training on the training data).
- **Database & Logging Infrastructure:** Efficient storage of complex training logs and weight matrices using structured HDF5 and SQLite storage.

---

## Project Structure

```text
.
├── data/                   # Data handling and storage
│   ├── dataloaders.py      # PyTorch Dataset/DataLoader configurations
│   ├── h5pydb.py           # Saving arrays/training logs in .h5 format
│   ├── sqlitedb.py         # Relational tracking of metadata
│   ├── MNIST/              # Raw MNIST data
│   └── training_results/   # Saved logs & checkpoints from model runs
│
├── estimator/              # Statistical estimators for α-stable distributions
│   ├── stable_estimators.py# Python implementations (e.g., Hill Estimator) for tail index
│   └── fit_stable.R        # R script for precise stable fits via McCulloch / Nolan
│
├── Grokking/               # Isolation and tests regarding the "grokking" phenomenon
│   ├── simple_models.py    # Minimalist models for grokking
│   └── simple_grokking_demo.ipynb
│
├── model_analytics/        # Jupyter Notebooks for post-training evaluation
│   ├── loss_surface.ipynb  # Test for Visualization of the loss surface (not used in final submission) 
│   ├── MNIST/ & CIFAR/     # Specific analyses for different architectures (FC3 focused; FC10 experiments paused due to compute limitations)
│   └── model_analytics.ipynb
│
├── model_training/         # Training scripts and pipeline
│   ├── training_pipeline.ipynb # Main pipeline for training the models
│   └── MNIST/ & CIFAR/     # Configurations for different datasets and architectures
│
├── Plots/                  # Evaluation and visualization results 
│
├── utils/                  # Helper functions
│   ├── analytics.py        # Metrics and analysis tools
│   ├── optimreg.py         # Custom optimizers & regularizers (e.g., L1, L2, Hill-regularizer)
│   └── plots.py            # Generation of consistent and clean plots
│
├── depr/                   # (Deprecated) Outdated scripts and earlier test runs
│
├── networks.py             # Base definitions for neural network architectures (PyTorch)
├── stable_fitting_eval.ipynb # Evaluation of alpha estimators for different weight matrices
├── regularizers.ipynb      # Experiments with the Hill-regularizer vs L1/L2
└── phase_diagram.ipynb     # Test for phase diagrams (not used in final submission) 
```

---

## Technical Specifications & Requirements

The project requires Python 3 (recommended: >= 3.8) and the following core libraries:
- `torch` & `torchvision` (PyTorch for the deep learning backend)
- `scipy` & `numpy` (for statistical functions, especially `scipy.stats.levy_stable`)
- `h5py` (for efficient matrix storage)
- `matplotlib` & `seaborn` (for data visualization)
- **R (optional)**: If the R script (`fit_stable.R`) is used for high-precision alpha estimation, an R environment including relevant packages (like `libstableR` or similar) is required.




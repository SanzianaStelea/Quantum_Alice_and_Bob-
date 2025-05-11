# ETH Quantum Hackathon 2025 — Alice & Bob Challenge

This repository contains our solution to the Alice & Bob quantum state tomography challenge. We focus on reconstructing quantum states from Wigner function measurements, using both analytical tools and machine learning methods for denoising and analysis.

---

## 🔧 Structure

| File/Folder | Description |
|-------------|-------------|
| `Challenge.md` | Task description (copied from official challenge) |
| `ETH_Hackathon_Challenge.pdf` | PDF version of the challenge |
| `TheoryBackground.md` | Notes on the Wigner function, parity measurements, tomography |
| `README.md` | You’re here. Summary of the approach |
| `task1a.ipynb` | Task A: Generate and visualize Wigner functions for various quantum states |
| `taskB.ipynb` | Task 1B + C: Quantum state tomography via convex optimization |
| `wigner_to_density_matrix_and_analyse.ipynb` | Final reconstruction + fidelity analysis |
| `trained_fidelities.ipynb` | Fidelity trends for the trained DAE |
| `fit_acceleration.ipynb` | Optimizations for faster reconstruction |
| `experimental_data_processing.ipynb` | Handling and smoothing experimental Wigner datasets |
| `exercise2.ipynb` | Ex 2A |
| `DenoisingAE.ipynb` | Denoising autoencoder for improving noisy Wigner functions |
| `data_generation.ipynb` | Simulation of noisy training data 201x201 |
| `data_generation_32.ipynb` | Variant with 32×32 resolution |
| `conv_res_mlp_dae_32x32.pth` | Trained PyTorch model for denoising |
| `loss_conv_res_mlp_dae_32x32.png` | DAE training loss plot |
| `wigner_train_32/` | Folder with training data |
| `results_experimental_data/` | Smoothed/processed Wigner images from real measurements |

---

## 🧠 Highlights

- **State reconstruction** via displaced parity measurements
- **Convex optimization** using CVXPY for maximum-likelihood tomography
- **Fidelity evaluation** vs. ground-truth states (Fock, cat, coherent)
- **Denoising autoencoder** for smoothing experimental Wigner maps
- **Interactive plots** of Wigner functions, diagonals, and eigenvalue distributions

---

## 📈 Dependencies

- `dynamiqs`, `jax`, `cvxpy`, `numpy`, `scipy`, `matplotlib`, `torch`

---

## ✅ Results

- Achieved >99% fidelity on even cat states with appropriate Hilbert dimension
- Visualized degradation from noise and over-/under-truncation
- Developed toolchain for analyzing fidelity sensitivity to displacement α and truncation

---

## ✍️ Authors

ETH Zurich Scoubidou team — Sânziana & Stefan

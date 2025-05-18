# ETH Quantum Hackathon 2025 — Alice & Bob Challenge

This repository contains our solution to the Alice & Bob quantum state tomography challenge. We focus on reconstructing quantum states from Wigner function measurements, using both analytical tools and machine learning methods for denoising and analysis.

---

## 🔧 Structure

| File/Folder | Description |
|-------------|-------------|
| `task1a.ipynb` | Task A: Generate and visualize Wigner functions for various quantum states |
| `taskB.ipynb` | Task B+C: Quantum state tomography via convex optimization |
| `experimental_data_processing.ipynb` | Handling and smoothing experimental Wigner datasets |
| `exercise2.ipynb` | Initial experiments & sanity checks |
| `wigner_to_density_matrix_and_analyse.ipynb` | Fidelity scan over Gaussian kernels |
| `DenoisingAE.ipynb` | Denoising autoencoder for improving noisy Wigner functions |
| `data_generation.ipynb` | Generation of noisy training data for the encoder 200x200 |
| `data_generation_32.ipynb` | Variant with 32×32 resolution |
| `trained_fidelities.ipynb` | Testing how using the developed ML-based denoiser improves the reconstruction fidelity for fock and coherent states |
| `fit_acceleration.ipynb` | First try at ML based optimizations for faster reconstruction |
| `conv_res_mlp_dae_32x32.pth` | Trained PyTorch model for denoising |
| `wigner_train_32/` | Folder with training data |

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

- Achieved >99% fidelity on flawless simulated data
- Studied how fidelity scales for different types of quantum states
- Visualized degradation from noise and over-/under-truncation
- Developed ML based denoiser 

---

## ✍️ Authors

ETH Zurich Scoubidou team — Sânziana & Stefan

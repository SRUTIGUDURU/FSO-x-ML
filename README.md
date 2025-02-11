# **FSO x ML**

**Free-Space Optics (FSO) meets Machine Learning (ML)** — a project exploring the impact of turbulence on Optical Vortex (OAM) modes and leveraging computational techniques to analyze and mitigate distortions in FSO communication systems.

## **Overview**

Free-Space Optical (FSO) communication is a wireless technology that uses laser beams to transmit data through the atmosphere. However, atmospheric turbulence introduces phase distortions, affecting signal integrity — especially for Orbital Angular Momentum (OAM) modes. This project simulates turbulence effects, applies attenuation models, and visualizes OAM fields to study their resilience in turbulent environments.

## **Features**

-   **OAM Field Simulation** – Generates helical wavefronts with topological charge `l`.
-   **Kolmogorov Turbulence Modeling** – Simulates phase distortions due to atmospheric turbulence.
-   **Attenuation Effects** – Applies a Gaussian radial attenuation to model beam propagation losses.
-   **Field Visualization** – Plots phase and intensity distributions for analysis.
-   **Customizable Parameters** – Allows user-defined OAM charge, grid size, turbulence strength, and attenuation factors.

## **Installation**

Ensure you have Python installed, then install the dependencies:

```bash
pip install numpy matplotlib
```

### Installing JAX (for Apple Silicon or GPU acceleration)

If you’re using an Apple Mac with an M1/M2/M3 chip and want to utilize the Metal GPU backend:

```bash
pip install jax jaxlib –no-cache-dir
```

For CUDA (NVIDIA GPU) acceleration, install JAX with CUDA support (ensure CUDA and cuDNN are properly set up):

```bash
pip install jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For CPU-only usage:

```bash
pip install jax jaxlib
```

### Installing CuPy (for NVIDIA GPU Acceleration)

If you have an NVIDIA GPU with CUDA installed, install CuPy for GPU-accelerated computations:

For CUDA 12 (Latest)

```bash
pip install cupy-cuda12x
```

For CUDA 11

```bash
pip install cupy-cuda11x
```

### Clone the repository:

Run this code to clone the repository

```bash
git clone https://github.com/Aravind-Sathesh/FSO-x-ML.git
cd FSO-x-ML
```

## **Usage**

Run the main script and enter the required parameters when prompted:

```bash
python main.py
```

The script will:

1. Generate an OAM field with a given topological charge.
2. Apply Gaussian attenuation based on the chosen factor.
3. Introduce Kolmogorov or Von Karmen turbulence to simulate atmospheric distortions.
4. Display phase and intensity distributions of the field.

Plots will be saved automatically.

## **Contributors**

-   **Sruti Guduru** – Developer
-   **Aravind Sathesh** – Developer

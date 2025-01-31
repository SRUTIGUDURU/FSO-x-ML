# **FSO x ML**

**Free-Space Optics (FSO) meets Machine Learning (ML)** — a project exploring the impact of turbulence on Optical Vortex (OAM) modes and leveraging computational techniques to analyze and mitigate distortions in FSO communication systems.

## **Overview**

Free-Space Optical (FSO) communication is a wireless technology that uses laser beams to transmit data through the atmosphere. However, atmospheric turbulence introduces phase distortions, affecting signal integrity — especially for Orbital Angular Momentum (OAM) modes. This project simulates turbulence effects, applies attenuation models, and visualizes OAM fields to study their resilience in turbulent environments.

## **Features**

- **OAM Field Simulation** – Generates helical wavefronts with topological charge `l`.  
- **Kolmogorov Turbulence Modeling** – Simulates phase distortions due to atmospheric turbulence.  
- **Attenuation Effects** – Applies a Gaussian radial attenuation to model beam propagation losses.  
- **Field Visualization** – Plots phase and intensity distributions for analysis.  
- **Customizable Parameters** – Allows user-defined OAM charge, grid size, turbulence strength, and attenuation factors.

## **Installation**
Ensure you have Python installed, then install the dependencies:

```bash
pip install numpy matplotlib
```

Clone the repository:

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
1. Generate an OAM field with a given topological charge `l`.  
2. Apply Gaussian attenuation based on the chosen factor.  
3. Introduce Kolmogorov turbulence to simulate atmospheric distortions.  
4. Display phase and intensity distributions of the field.  

Plots will be saved automatically if `save_plots=True` in `field_all_plot()`.

## **Contributors**

- **Sruti Guduru** – Developer
- **Aravind Sathesh** – Developer

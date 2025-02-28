import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jax_rand
import matplotlib.pyplot as plt

xp = jnp  # JAX for computations
random_generator = jax_rand  # JAX random functions

def simulate_oam(l, grid_size, r_max):
    """Generates an OAM field with topological charge l."""
    x = xp.linspace(-1, 1, grid_size) * r_max
    y = xp.linspace(-1, 1, grid_size) * r_max
    X, Y = xp.meshgrid(x, y)
    phase = xp.arctan2(Y, X)  # Corrected phase calculation
    return xp.exp(1j * l * phase)

def kolmogorov_turbulence(grid_size, r0, distance, key, model="kolmogorov"):
    k = xp.fft.fftfreq(grid_size, d=1.0)
    Kx, Ky = xp.meshgrid(k, k, indexing='ij')
    K = xp.sqrt(Kx**2 + Ky**2)
    K = xp.where(K == 0, 1e-10, K)

    if model == "kolmogorov":
        spectrum = (K ** (-11 / 3)) * xp.exp(- (K / (1 / r0)) ** 2)
    elif model == "von_karman":
        L0 = 10
        spectrum = (K ** (-11 / 3)) * xp.exp(- (K * L0) ** 2)
    else:
        raise ValueError("Invalid turbulence model.")

    turbulence_strength = 0.1 * distance

    key, subkey = jax_rand.split(key)
    random_values = jax_rand.uniform(subkey, (grid_size, grid_size), minval=-xp.pi, maxval=xp.pi)
    random_phase = xp.exp(1j * random_values)
    turbulence_screen = xp.fft.ifft2(xp.fft.ifftshift(xp.sqrt(spectrum) * random_phase)).real

    turbulence_screen /= xp.max(xp.abs(turbulence_screen))
    turbulence_screen *= turbulence_strength

    key, subkey = jax_rand.split(key)
    turbulence_screen += 0.05 * jax_rand.normal(subkey, (grid_size, grid_size))

    amplitude_variation = 1 + 0.3 * turbulence_screen
    return turbulence_screen, amplitude_variation

def attenuation(grid_size, attenuation_factor, r_max, distance, key):
    """Generates an improved attenuation mask with smoother spatial variations."""
    x = xp.linspace(-1, 1, grid_size) * r_max
    y = xp.linspace(-1, 1, grid_size) * r_max
    X, Y = xp.meshgrid(x, y)
    r = xp.sqrt(X**2 + Y**2) / r_max

    # Base attenuation using an exponential decay function
    base_attenuation = xp.exp(-attenuation_factor * distance * (r ** 2.5))

    # Generate spatially correlated noise for smoother variation
    key, subkey = jax_rand.split(key)
    noise = jax_rand.normal(subkey, (grid_size, grid_size))
    
    # Apply a Gaussian filter in frequency domain with a lower cutoff frequency
    kx = xp.fft.fftfreq(grid_size, d=1.0)
    ky = xp.fft.fftfreq(grid_size, d=1.0)
    KX, KY = xp.meshgrid(kx, ky)
    K = xp.sqrt(KX**2 + KY**2)
    K = xp.where(K == 0, 1e-10, K)  # Prevent division by zero

    noise_spectrum = xp.fft.fft2(noise) * xp.exp(-0.01 * (K ** 2))  # Smoother Gaussian filter
    correlated_noise = xp.real(xp.fft.ifft2(noise_spectrum))

    # Normalize and scale the noise
    correlated_noise /= xp.std(correlated_noise)
    correlated_noise *= 0.05  # Reduced strength for smoother attenuation variations

    # Combine base attenuation with smoother noise variations
    attenuation_mask = base_attenuation * (1 + correlated_noise)
    attenuation_mask = xp.clip(attenuation_mask, 0, 1)  # Ensure values stay in [0,1] range

    return attenuation_mask

def simulate_oam_toroid(l, grid_size, r_max, r_peak, width):
    """Generates an OAM field with a toroidal intensity profile."""
    x = xp.linspace(-1, 1, grid_size) * r_max
    y = xp.linspace(-1, 1, grid_size) * r_max
    X, Y = xp.meshgrid(x, y)
    r = xp.sqrt(X**2 + Y**2)

    phase = xp.arctan2(Y, X)
    toroidal_envelope = xp.exp(-((r - r_peak) / width) ** 2)
    asymmetry_factor = 1 + 0.2 * xp.sin(2 * xp.pi * Y / r_max)

    oam_field = toroidal_envelope * xp.exp(1j * l * phase) * asymmetry_factor
    return oam_field

def plot_field(field, r_max, title, cmap='hsv', add_noise=False, noise_level=0.1, normalize_intensity=True):
    """Plots the phase and intensity of a field, with optional noise and normalization."""
    field = field.block_until_ready()
    field_cpu = np.array(field)
    
    if add_noise:
        # Add random noise to the phase
        phase_noise = noise_level * np.random.uniform(-np.pi, np.pi, field_cpu.shape)
        field_cpu = np.abs(field_cpu) * np.exp(1j * (np.angle(field_cpu) + phase_noise))
        
        # Add random noise to the intensity
        intensity_noise = noise_level * np.random.normal(0, 1, field_cpu.shape)
        field_cpu = (np.abs(field_cpu) + intensity_noise) * np.exp(1j * np.angle(field_cpu))
    
    if normalize_intensity:
        intensity = np.abs(field_cpu) ** 2
        intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        field_cpu = np.sqrt(intensity) * np.exp(1j * np.angle(field_cpu))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(np.angle(field_cpu), extent=[-r_max, r_max, -r_max, r_max], cmap='twilight', interpolation='bicubic')
    plt.title(title + " (Phase)")
    plt.colorbar(label="Phase (radians)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.draw()

    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(field_cpu) ** 2, extent=[-r_max, r_max, -r_max, r_max], cmap='hot', interpolation='bicubic')
    plt.title(title + " (Intensity)")
    plt.colorbar(label="Intensity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.draw()

def generate_all_variations(l, grid_size, r_max, r0, attenuation_factor, distance, r_peak, width, turbulence_model="kolmogorov", save_path="saved_fields"):
    """Generates and saves all possible OAM field variations for spherical and toroidal modes."""
    save_path = os.path.join(os.getcwd(), save_path)
    os.makedirs(save_path, exist_ok=True)
    
    key = jax_rand.PRNGKey(0)
    
    print("Generating standard OAM field...")
    oam_sphere = simulate_oam(l, grid_size, r_max)
    plot_field(oam_sphere, r_max, "Spherical OAM Mode", add_noise=True, noise_level=0.1)
    
    print("Generating toroidal OAM field...")
    oam_toroid = simulate_oam_toroid(l, grid_size, r_max, r_peak, width)
    plot_field(oam_toroid, r_max, "Toroidal OAM Mode", add_noise=True, noise_level=0.1)
    
    print("Generating turbulence...")
    turbulence_screen, amplitude_variation = kolmogorov_turbulence(grid_size, r0, distance, key, model=turbulence_model)
    plot_field(xp.exp(1j * 2 * xp.pi * turbulence_screen), r_max, "Turbulence Phase", add_noise=True, noise_level=0.1)
    
    print("Generating attenuation field...")
    attenuation_field = attenuation(grid_size, attenuation_factor, r_max, distance, key)
    plot_field(attenuation_field, r_max, "Attenuation Field", cmap='gray', add_noise=True, noise_level=0.1)
    
    print("Applying distortions...")
    turbulent_sphere = oam_sphere * xp.exp(1j * 2 * xp.pi * turbulence_screen) * amplitude_variation
    turbulent_toroid = oam_toroid * xp.exp(1j * 2 * xp.pi * turbulence_screen) * amplitude_variation
    
    plot_field(turbulent_sphere, r_max, "Spherical OAM with Turbulence", add_noise=True, noise_level=0.1)
    plot_field(turbulent_toroid, r_max, "Toroidal OAM with Turbulence", add_noise=True, noise_level=0.1)
    
    attenuated_sphere = turbulent_sphere * attenuation_field
    attenuated_toroid = turbulent_toroid * attenuation_field
    
    plot_field(attenuated_sphere, r_max, "Spherical OAM with Turbulence & Attenuation", add_noise=True, noise_level=0.1)
    plot_field(attenuated_toroid, r_max, "Toroidal OAM with Turbulence & Attenuation", add_noise=True, noise_level=0.1)
    
    np.savez_compressed(os.path.join(save_path, "oam_variations.npz"),
                        sphere_clean=np.abs(np.array(oam_sphere)),
                        toroid_clean=np.abs(np.array(oam_toroid)),
                        sphere_turbulent=np.abs(np.array(turbulent_sphere)),
                        toroid_turbulent=np.abs(np.array(turbulent_toroid)),
                        sphere_att=np.abs(np.array(attenuated_sphere)),
                        toroid_att=np.abs(np.array(attenuated_toroid)))
    
    print(f"All data saved in '{save_path}' directory.")

def main():
    l, grid_size, r_max, r0, attenuation_factor, distance = 2, 2500, 1, 0.02, 3, 10
    r_peak, width = 0.5, 0.1
    turbulence_model = "kolmogorov"
    
    generate_all_variations(l, grid_size, r_max, r0, attenuation_factor, distance, r_peak, width, turbulence_model)
    plt.show()

if __name__ == "__main__":
    main()

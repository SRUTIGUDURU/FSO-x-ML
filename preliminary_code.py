import os
import numpy as np
import jax.random as jax_rand
import matplotlib.pyplot as plt

# Auto-detect and select the best library
try:
    import cupy as xp  # Try CuPy first (CUDA GPU)
    backend = "CuPy (GPU)"
except ImportError:
    try:
        import jax.numpy as xp  # Try JAX (Apple Metal GPU)
        backend = "JAX (Metal GPU)"
    except ImportError:
        import numpy as xp  # Default to NumPy (CPU)
        backend = "NumPy (CPU)"

print(f"Using {backend} for computations.")


def simulate_oam(l, grid_size, r_max):
    """Dummy function to simulate an OAM field."""
    x = xp.linspace(-1, 1, grid_size) * r_max
    y = xp.linspace(-1, 1, grid_size) * r_max
    X, Y = xp.meshgrid(x, y)
    phase = xp.angle(X + 1j * Y)
    return xp.exp(1j * l * phase)


def kolmogorov_turbulence(grid_size, r0, model="kolmogorov"):
    """Generates turbulence screen based on Kolmogorov or von Kármán model."""
    k = np.fft.fftfreq(grid_size, d=1.0)
    Kx, Ky = xp.meshgrid(k, k)
    K = xp.sqrt(Kx**2 + Ky**2)
    K = xp.where(K == 0, 1e-10, K)

    if model == "kolmogorov":
        spectrum = (K ** (-11 / 3)) * xp.exp(- (K / (1 / r0)) ** 2)
    elif model == "von_karman":
        L0 = 10  # Outer scale of turbulence
        spectrum = (K ** (-11 / 3)) * xp.exp(- (K * L0) ** 2)
    else:
        raise ValueError(
            "Invalid turbulence model. Use 'kolmogorov' or 'von_karman'.")

    if backend == "JAX (Metal GPU)":
        key, subkey = jax_rand.split(jax_rand.PRNGKey(0))
        random_values = jax_rand.uniform(subkey, (grid_size, grid_size))
    else:
        random_values = xp.random.rand(grid_size, grid_size)
    random_phase = xp.exp(1j * 2 * xp.pi * random_values)
    turbulence_screen = xp.fft.ifft2(
        xp.fft.ifftshift(xp.sqrt(spectrum) * random_phase)
    ).real
    turbulence_screen /= xp.max(xp.abs(turbulence_screen))
    amplitude_variation = 1 + 0.2 * turbulence_screen
    return turbulence_screen, amplitude_variation


def attenuation(grid_size, attenuation_factor, r_max):
    """Generates an attenuation mask based on radial distance."""
    x = xp.linspace(-1, 1, grid_size) * r_max
    y = xp.linspace(-1, 1, grid_size) * r_max
    X, Y = xp.meshgrid(x, y)
    radial_distance = xp.sqrt(X**2 + Y**2)
    attenuation = xp.exp(-attenuation_factor * (radial_distance / r_max) ** 2)

    # Adding randomness
    if backend == "JAX (Metal GPU)":
        key, subkey = jax_rand.split(jax_rand.PRNGKey(1))  # Fix key issue
        random_values = jax_rand.uniform(subkey, (grid_size, grid_size))
    else:
        random_values = xp.random.rand(grid_size, grid_size)
    attenuation *= (0.9 + 0.1 * random_values)
    return attenuation


def plot_field(field, r_max, title, cmap='hsv'):
    """Plots phase and intensity of the given complex field."""
    field_cpu = np.array(field)  # Works for JAX, NumPy, and CuPy

    plt.figure(figsize=(6, 6))
    plt.imshow(np.angle(field_cpu),
               extent=[-r_max, r_max, -r_max, r_max], cmap=cmap, interpolation='bilinear')
    plt.title(title + " (Phase)")
    plt.colorbar(label="Phase (radians)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=False)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(field_cpu) ** 2,
               extent=[-r_max, r_max, -r_max, r_max], cmap='hot', interpolation='bilinear')
    plt.title(title + " (Intensity)")
    plt.colorbar(label="Intensity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show(block=False)


def generate_and_save_data(l, grid_size, r_max, r0, attenuation_factor, turbulence_model="kolmogorov", save_path="saved_fields"):
    """Generates and saves OAM fields with different distortions."""

    save_path = os.path.join(os.getcwd(), save_path)
    os.makedirs(save_path, exist_ok=True)

    print("Generating OAM field...")
    oam_field = simulate_oam(l, grid_size, r_max)

    print("Generating turbulence...")
    turbulence_screen, amplitude_variation = kolmogorov_turbulence(
        grid_size, r0, model=turbulence_model)

    print("Generating attenuation field...")
    attenuation_field = attenuation(grid_size, attenuation_factor, r_max)

    print("Applying distortions...")
    attenuated_oam = oam_field * attenuation_field
    turbulent_oam = oam_field * \
        xp.exp(1j * 2 * xp.pi * turbulence_screen) * amplitude_variation
    turbulent_attenuated_oam = turbulent_oam * attenuation_field

    np.savez_compressed(os.path.join(save_path, "oam_data.npz"),
                        oam_clean=np.abs(np.array(oam_field)),
                        oam_turbulent=np.abs(np.array(turbulent_oam)),
                        oam_attenuated=np.abs(np.array(attenuated_oam)),
                        oam_turb_att=np.abs(np.array(turbulent_attenuated_oam)))

    print(f"Data saved in '{save_path}' directory.")

    # Plot the fields
    plot_field(oam_field, r_max, "OAM Mode")
    plot_field(turbulent_oam, r_max, "OAM with Turbulence")
    plot_field(attenuated_oam, r_max, "OAM with Attenuation")
    plot_field(turbulent_attenuated_oam, r_max,
               "OAM with Turbulence & Attenuation")

    return oam_field, attenuated_oam, turbulent_oam, turbulent_attenuated_oam


def main():
    l = 1  # OAM Mode
    grid_size = 256  # Grid resolution
    r_max = 1  # Max radius
    r0 = 0.1  # Fried parameter (turbulence strength)
    attenuation_factor = 2.0  # Attenuation strength
    turbulence_model = "von_karman"  # Use 'kolmogorov' or 'von_karman'

    generate_and_save_data(l, grid_size, r_max, r0,
                           attenuation_factor, turbulence_model)
    plt.show()


if __name__ == "__main__":
    main()

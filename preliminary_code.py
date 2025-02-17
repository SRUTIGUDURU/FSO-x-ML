import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jax_rand
import matplotlib.pyplot as plt

try:
    xp = jnp  # JAX for computations
    random_generator = jax_rand  # JAX random functions
    print("Using JAX for computations.")
except ImportError:
    xp = np  # Fallback to NumPy
    random_generator = np.random
    print("JAX not found. Falling back to NumPy.")


def simulate_oam(l, grid_size, r_max):
    """Generates an OAM field with topological charge l."""
    x = xp.linspace(-1, 1, grid_size) * r_max
    y = xp.linspace(-1, 1, grid_size) * r_max
    X, Y = xp.meshgrid(x, y)
    phase = xp.angle(X + 1j * Y)
    return xp.exp(1j * l * phase)


def kolmogorov_turbulence(grid_size, r0, distance, key, model="kolmogorov"):
    """Generates a turbulence phase screen with stronger effects over distance."""
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

    turbulence_strength = 0.1 * distance  # Increase phase distortion over distance

    # Generate JAX random values
    key, subkey = jax_rand.split(key)
    random_values = jax_rand.uniform(subkey, (grid_size, grid_size))

    random_phase = xp.exp(1j * 2 * xp.pi * random_values)
    turbulence_screen = xp.fft.ifft2(
        xp.fft.ifftshift(xp.sqrt(spectrum) * random_phase)
    ).real

    # Normalize & amplify effects
    turbulence_screen /= xp.max(xp.abs(turbulence_screen))
    turbulence_screen *= turbulence_strength

    # Add Gaussian noise
    key, subkey = jax_rand.split(key)
    turbulence_screen += 0.05 * jax_rand.normal(subkey, (grid_size, grid_size))

    amplitude_variation = 1 + 0.3 * turbulence_screen  # Increased effect
    return turbulence_screen, amplitude_variation


def attenuation(grid_size, attenuation_factor, r_max, distance, key):
    """Generates an attenuation mask with stronger effects over distance."""
    x = xp.linspace(-1, 1, grid_size) * r_max
    y = xp.linspace(-1, 1, grid_size) * r_max
    X, Y = xp.meshgrid(x, y)
    radial_distance = xp.sqrt(X**2 + Y**2)

    # Exponential decay with distance
    attenuation = xp.exp(-attenuation_factor *
                         (radial_distance / r_max) ** 2 * distance)

    # Generate JAX random values
    key, subkey = jax_rand.split(key)
    random_values = jax_rand.uniform(subkey, (grid_size, grid_size))

    attenuation *= (0.8 + 0.2 * random_values)  # Stronger randomness
    return attenuation


def plot_field(field, r_max, title, cmap='hsv'):
    """Plots phase and intensity of the given complex field."""
    field_cpu = np.array(field)  # Convert for visualization

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


def generate_and_save_data(l, grid_size, r_max, r0, attenuation_factor, distance, turbulence_model="kolmogorov", save_path="saved_fields"):
    """Generates OAM fields with turbulence & attenuation effects."""

    save_path = os.path.join(os.getcwd(), save_path)
    os.makedirs(save_path, exist_ok=True)

    key = jax_rand.PRNGKey(0)  # JAX random key

    print("Generating OAM field...")
    oam_field = simulate_oam(l, grid_size, r_max)

    print("Generating turbulence...")
    turbulence_screen, amplitude_variation = kolmogorov_turbulence(
        grid_size, r0, distance, key, model=turbulence_model)

    print("Generating attenuation field...")
    attenuation_field = attenuation(
        grid_size, attenuation_factor, r_max, distance, key)

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
    distance = 2  # Distance in meters
    turbulence_model = "von_karman"  # 'kolmogorov' or 'von_karman'

    generate_and_save_data(l, grid_size, r_max, r0,
                           attenuation_factor, distance, turbulence_model)
    plt.show()


if __name__ == "__main__":
    main()

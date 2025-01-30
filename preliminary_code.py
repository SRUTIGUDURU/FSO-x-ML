import numpy as np
import matplotlib.pyplot as plt

def kolmogorov_turbulence(grid_size, r0):
    """Generates a Kolmogorov turbulence phase screen."""
    delta = 1.0
    k = np.fft.fftfreq(grid_size, delta)
    Kx, Ky = np.meshgrid(k, k)
    K = np.sqrt(Kx**2 + Ky**2 + 1e-10)
    kolmogorov_spectrum = (K**(-11/3)) * np.exp(-K**2 * r0**2)
    random_phase = (np.random.normal(size=(grid_size, grid_size)) +
                    1j * np.random.normal(size=(grid_size, grid_size)))
    turbulence_screen = np.fft.ifft2(np.sqrt(kolmogorov_spectrum) * random_phase).real
    return turbulence_screen

def attenuation(grid_size, attenuation_factor, r_max):
    """Applies radial attenuation to the field."""
    x = np.linspace(-r_max, r_max, grid_size)
    y = np.linspace(-r_max, r_max, grid_size)
    X, Y = np.meshgrid(x, y)
    radial_distance = np.sqrt(X**2 + Y**2)
    attenuation = np.exp(-attenuation_factor * radial_distance**2)
    return attenuation

def simulate_oam(l, grid_size, r_max):
    """Generates an OAM field with topological charge `l`."""
    x = np.linspace(-r_max, r_max, grid_size)
    y = np.linspace(-r_max, r_max, grid_size)
    X, Y = np.meshgrid(x, y)
    radial_distance = np.sqrt(X**2 + Y**2)
    azimuthal_angle = np.arctan2(Y, X)
    helical_phase = np.exp(1j * l * azimuthal_angle)
    gaussian_envelope = np.exp((-radial_distance**2) / ((r_max / 3) ** 2))
    oam_mode = helical_phase * gaussian_envelope
    return oam_mode

def plot_field(field, r_max, title_phase, title_intensity, figure_number, save_filename=None):
    """Plots the phase and intensity of a field and saves it if needed."""
    plt.figure(figure_number, figsize=(20, 10))

    # Phase plot
    plt.subplot(1, 2, 1)
    plt.imshow(np.angle(field), extent=[-r_max, r_max, -r_max, r_max], cmap='hsv', interpolation='bilinear')
    plt.colorbar(label="Phase (radians)")
    plt.title(title_phase)

    # Intensity plot
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(field)**2, extent=[-r_max, r_max, -r_max, r_max], cmap='hot', interpolation='bilinear')
    plt.colorbar(label="Intensity")
    plt.title(title_intensity)

    plt.tight_layout()

    # Save before showing
    if save_filename:
        plt.savefig(save_filename, dpi=300)
        print(f"Saved plot as {save_filename}")

def field_all_plot(save_plots=False):
    """Runs OAM simulation, applies attenuation and turbulence, and plots results."""
    a = int(input("Enter topological charge (l): "))
    b = int(input("Enter grid size (positive integer): "))
    c = int(input("Enter maximum radius (r_max): "))
    r0 = float(input("Enter Fried parameter (r0, positive value): "))
    attenuation_factor = float(input("Enter attenuation factor (positive value): "))

    """Simulating the OAM field"""
    oam_field = simulate_oam(a, b, c)
    plot_field(oam_field, c, f"OAM Phase (l={a})", f"OAM Intensity (l={a})",
               figure_number=1, save_filename="oam_field.png" if save_plots else None)

    """Applying attenuation"""
    attenuated_field = oam_field * attenuation(b, attenuation_factor, c)
    plot_field(attenuated_field, c, "Attenuated OAM Phase", "Attenuated OAM Intensity",
               figure_number=2, save_filename="attenuated_field.png" if save_plots else None)

    """Applying Kolmogorov turbulence"""
    turbulence_screen = kolmogorov_turbulence(b, r0)
    turbulent_field = attenuated_field * np.exp(1j * turbulence_screen)
    plot_field(turbulent_field, c, "Turbulent & Attenuated OAM Phase", "Turbulent & Attenuated OAM Intensity",
               figure_number=3, save_filename="turbulent_attenuated_field.png" if save_plots else None)

    """Show all figures at the end"""
    plt.show()

    return oam_field, attenuated_field, turbulent_field

# Run with plot saving enabled
field_all_plot(save_plots=True)

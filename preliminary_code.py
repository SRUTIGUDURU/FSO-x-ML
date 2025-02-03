import numpy as np
import matplotlib.pyplot as plt

def kolmogorov_turbulence(grid_size, r0):
    k = np.fft.fftfreq(grid_size, d=1.0)
    Kx, Ky = np.meshgrid(k, k)
    K = np.sqrt(Kx**2 + Ky**2)
    K[K == 0] = 1e-10

    kolmogorov_spectrum = (K**(-11/3)) * np.exp(- (K / (1/r0))**2)
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(grid_size, grid_size))
    turbulence_screen = np.fft.ifft2(
        np.fft.ifftshift(np.sqrt(kolmogorov_spectrum) * random_phase)
    ).real

    return turbulence_screen


def attenuation(grid_size, attenuation_factor, r_max):
    x = np.linspace(-1, 1, grid_size) * r_max
    y = np.linspace(-1, 1, grid_size) * r_max
    X, Y = np.meshgrid(x, y)
    radial_distance = np.sqrt(X**2 + Y**2)
    attenuation = np.exp(-attenuation_factor * (radial_distance / r_max)**2)
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

def plot_field_separate(field, r_max, title_phase, title_intensity):
    plt.figure(figsize=(6, 6))
    plt.imshow(np.angle(field), extent=[-r_max, r_max, -r_max, r_max], cmap='hsv', interpolation='bilinear')
    plt.title(title_phase)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Phase (radians)")
    plt.show(block=False)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(field)**2, extent=[-r_max, r_max, -r_max, r_max], cmap='hot', interpolation='bilinear')
    plt.title(title_intensity)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Intensity")
    plt.show(block=False)

def field_all_plot():
    a = int(input("Enter topological charge (l): "))
    b = int(input("Enter grid size (positive integer): "))
    c = int(input("Enter maximum radius (r_max): "))
    r0 = float(input("Enter Fried parameter (r0, positive value): "))
    attenuation_factor = float(input("Enter attenuation factor (positive value): "))
    
    oam_field = simulate_oam(a, b, c)
    turbulence_screen = kolmogorov_turbulence(b, r0)
    attenuation_field = attenuation(b, attenuation_factor, c)
    
    plot_field_separate(oam_field, c, "OAM Phase", "OAM Intensity")
    plot_field_separate(np.exp(1j * 2 * np.pi * turbulence_screen), c, "Turbulence Phase", "Turbulence Intensity")
    plot_field_separate(attenuation_field, c, "Attenuation Phase", "Attenuation Intensity")
    
    attenuated_oam = oam_field * attenuation_field
    turbulent_oam = oam_field * np.exp(1j * 2 * np.pi * turbulence_screen)
    turbulent_attenuated_oam = turbulent_oam * attenuation_field
    
    plot_field_separate(attenuated_oam, c, "OAM with Attenuation Phase", "OAM with Attenuation Intensity")
    plot_field_separate(turbulent_oam, c, "OAM with Turbulence Phase", "OAM with Turbulence Intensity")
    plot_field_separate(turbulent_attenuated_oam, c, "OAM with Turbulence & Attenuation Phase", "OAM with Turbulence & Attenuation Intensity")
    
    plt.show()
    
    return oam_field, attenuated_oam, turbulent_oam, turbulent_attenuated_oam

if __name__ == "__main__":
    field_all_plot()


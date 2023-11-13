import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import sawtooth


# Streamlit app title
st.title('Nyquist-Shannon Theorem Demonstration')

# Signal-related inputs
st.sidebar.header('Signal Parameters')
f1_ratio = st.sidebar.slider('Frequency Ratio (f1/f2)', 0.1, 20.0, 0.5, help="Ratio of Frequency 1 to Frequency 2")
st.sidebar.write("Nyquist freq at f1/f2 = 2")
waveform = st.sidebar.selectbox('Input Waveform', ['Sine', 'Sawtooth', 'Complex'], index=0)
derived_signal_type = st.sidebar.selectbox('Derived Signal Type', ['Spline', 'Linear', 'Sine'], index=1)

# Design-related inputs
st.sidebar.header('Design Parameters')
first_line_color = st.sidebar.color_picker('First Line Color', '#008000')
first_line_width = st.sidebar.slider('First Line Width', 1, 5, 2)
second_line_color = st.sidebar.color_picker('Second Line Color', '#0000FF')
second_line_width = st.sidebar.slider('Second Line Width', 1, 5, 2)
sampling_point_color = st.sidebar.color_picker('Sampling Point Color', '#FF0000')
sampling_point_size = st.sidebar.slider('Sampling Point Size', 10, 100, 50)

# Function to generate the waveform
def generate_waveform(freq, t, wave_type):
    if wave_type == 'Sine':
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == 'Sawtooth':
        return sawtooth(2 * np.pi * freq * t)
    else:  # Complex multi-sinusoidal waveform
        return np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t)

# Create a time vector for the original signal
duration = 1
t = np.linspace(0, duration, 1000, endpoint=False)
f1 = 10  # fixed f1 (f2)
f2 = f1 * f1_ratio  # f1 (f1) based on the ratio
original_signal = generate_waveform(f1, t, waveform)

# Sampling and reconstruction
sampling_interval = 1 / f2
sample_points = np.arange(0, duration, sampling_interval)
sampled_signal = generate_waveform(f1, sample_points, waveform)

# Reconstructed signal
if derived_signal_type == 'Spline':
    reconstructor = CubicSpline(sample_points, sampled_signal)
elif derived_signal_type == 'Linear':
    reconstructor = interp1d(sample_points, sampled_signal, kind='linear', fill_value="extrapolate")
elif derived_signal_type == 'Polynomial':
    poly_order = min(len(sample_points) - 1, 3)  # Cap the polynomial order to avoid overfitting
    coefs = np.polyfit(sample_points, sampled_signal, poly_order)
    reconstructor = np.poly1d(coefs)
else:  # Sine
    reconstructor = lambda x: np.sin(2 * np.pi * f1 * x)

reconstructed_signal = reconstructor(t) if derived_signal_type != 'Sine' else reconstructor(t)

# Plotting
plt.figure(figsize=(15, 8))
plt.plot(t, original_signal, color=first_line_color, linewidth=first_line_width, label="Original Signal")
plt.plot(t, reconstructed_signal, color=second_line_color, linestyle='--', linewidth=second_line_width, label="Reconstructed Signal")
plt.scatter(sample_points, sampled_signal, color=sampling_point_color, s=sampling_point_size, label="Sampling Points")
plt.title("Nyquist-Shannon Theorem Demonstration")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
st.pyplot(plt)

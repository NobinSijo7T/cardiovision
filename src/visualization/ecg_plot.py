"""
CARDIOVISION - ECG Plotting
Visualizations for raw and filtered ECG signals.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_12_lead_ecg(signal: np.ndarray, sampling_rate: int, title: str = "12-Lead ECG", figsize=(15, 10)):
    """
    Plot 12-lead ECG in a standard clinical grid.
    
    Layout:
    I    aVR  V1  V4
    II   aVL  V2  V5
    III  aVF  V3  V6
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    # Mapping PTB-XL order to standard display order
    # PTB-XL: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
    
    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16)
    
    time_axis = np.arange(len(signal)) / sampling_rate
    
    for i in range(3):
        for j in range(4):
            lead_idx = j * 3 + i
            if lead_idx < signal.shape[1]:
                ax = axes[i, j]
                ax.plot(time_axis, signal[:, lead_idx], color='black', linewidth=1)
                ax.set_title(lead_names[lead_idx])
                ax.grid(True, which='both', linestyle='--', alpha=0.5)
                
                # Highlight zero line
                ax.axhline(0, color='red', linewidth=0.5, alpha=0.5)
                
    plt.tight_layout()
    return fig

def plot_signal_with_r_peaks(signal: np.ndarray, r_peaks: np.ndarray, sampling_rate: int, title: str = "ECG with Detected R-Peaks", lead_idx: int = 1, figsize=(12, 4)):
    """Plot a single ECG lead with marked R-peaks."""
    fig, ax = plt.subplots(figsize=figsize)
    
    time_axis = np.arange(len(signal)) / sampling_rate
    
    ax.plot(time_axis, signal[:, lead_idx], color='blue', label='ECG Signal', linewidth=1)
    
    if len(r_peaks) > 0:
        r_peak_times = r_peaks / sampling_rate
        r_peak_vals = signal[r_peaks, lead_idx]
        ax.plot(r_peak_times, r_peak_vals, 'ro', label='Detected R-Peaks')
        
    ax.set_title(f"{title} (Lead {lead_idx})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    return fig

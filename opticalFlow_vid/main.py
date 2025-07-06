import cv2
import numpy as np
from ix_iy_it_calc import calcIxIyIt
from scipy.ndimage import convolve
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def HS(img1, img2, alpha):
    gimg1 = cv2.GaussianBlur(img1, (5, 5), 0)
    gimg2 = cv2.GaussianBlur(img2, (5, 5), 0)

    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    Ix, Iy, It = calcIxIyIt(gimg1, gimg2)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                           [1 / 6, 0, 1 / 6],
                           [1 / 12, 1 / 6, 1 / 12]], float)

    iter_count = 0
    while True:
        iter_count += 1
        u_avg = convolve(u, avg_kernel)
        v_avg = convolve(v, avg_kernel)
        p = Ix * u_avg + Iy * v_avg + It
        d = alpha**2 + Ix**2 + Iy**2

        u = u_avg - Ix * (p / d)
        v = v_avg - Iy * (p / d)

        if iter_count > 50:
            break

    return u, v

def process_video(video_path, alpha):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(float)
    
    u_all, v_all = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        u, v = HS(prev_frame, curr_frame, alpha)
        u_all.append(u)
        v_all.append(v)
        prev_frame = curr_frame

    cap.release()

    return np.array(u_all), np.array(v_all)

def compute_temporal_fft(u_all, v_all):
    # Apply FFT
    u_fft = fft(u_all, axis=0)
    v_fft = fft(v_all, axis=0)

    # Power Spectral Density (PSD)
    u_psd = np.abs(u_fft)**2
    v_psd = np.abs(v_fft)**2

    # Sum over all pixels to get overall PSD
    psd = np.sum(u_psd, axis=(1, 2)) + np.sum(v_psd, axis=(1, 2))
    frequencies = np.fft.fftfreq(u_all.shape[0])

    # Filter to only positive frequencies
    pos_indices = frequencies >= 0
    psd = psd[pos_indices]
    frequencies = frequencies[pos_indices]
    u_fft = u_fft[pos_indices]
    v_fft = v_fft[pos_indices]

    return psd, frequencies, u_fft, v_fft


def plot_results(psd, frequencies, u_fft, v_fft):
    # Find peaks in the PSD
    peak_indices, _ = find_peaks(psd)
    
    # Sort peaks by their PSD values in descending order and get top 2
    significant_indices = peak_indices[np.argsort(psd[peak_indices])[-2:][::-1]]
    significant_freqs = frequencies[significant_indices]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(frequencies, psd)
    axes[0, 0].set_title("Power Spectral Density vs Frequency")
    axes[0, 0].set_xlabel("Frequency")
    axes[0, 0].set_ylabel("Power Spectral Density")
    axes[0, 0].scatter(significant_freqs, psd[significant_indices], color="red", label="Peaks")
    axes[0, 0].legend()

    for i, freq_idx in enumerate(significant_indices):
        u_mode = np.abs(u_fft[freq_idx])
        v_mode = np.abs(v_fft[freq_idx])

        ax = axes[0, i + 1]  
        im = ax.imshow(u_mode, cmap="jet")
        ax.set_title(f"X Mode (u) for Frequency {frequencies[freq_idx]:.2f}")
        fig.colorbar(im, ax=ax, orientation="vertical")

        ax = axes[1, i]  
        im = ax.imshow(v_mode, cmap="jet")
        ax.set_title(f"Y Mode (v) for Frequency {frequencies[freq_idx]:.2f}")
        fig.colorbar(im, ax=ax, orientation="vertical")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    video_path = r"/Users/akshatsrivastava/Desktop/VnV/bridgevideo.mp4"
    alpha = 25

    u_all, v_all = process_video(video_path, alpha)
    psd, frequencies, u_fft, v_fft = compute_temporal_fft(u_all, v_all)
    plot_results(psd, frequencies, u_fft, v_fft)

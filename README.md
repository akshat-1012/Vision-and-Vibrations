# Image-Space Modal Analysis for Video

A Python implementation of modal analysis techniques for extracting vibration modes from video sequences, inspired by the research paper "Image-Space Modal Bases for Plausible Manipulation of Objects in Video" by Abe Davis, Justin G. Chen, and Frédo Durand (MIT CSAIL).

## Overview

This project implements key algorithms for analyzing object dynamics in video by extracting modal bases from optical flow data. While the original paper focuses on interactive manipulation, this implementation concentrates on the modal extraction and analysis components.

## What's Implemented

### 1. **Canny Edge Detection** (From Scratch)
- **Gaussian Blur**: Noise reduction preprocessing
- **Gradient Calculation**: Using Sobel operators for edge detection
- **Non-Maximum Suppression**: Thin edges to single pixel width
- **Hysteresis Thresholding**: Robust edge linking with dual thresholds

### 2. **Horn-Schunck Optical Flow**
- **Spatial/Temporal Derivatives**: Calculates image gradients (Ix, Iy, It)
- **Iterative Optimization**: Solves optical flow equations with smoothness constraint
- **Flow Visualization**: Quiver plots showing motion vectors

### 3. **Modal Analysis**
- **Temporal FFT**: Frequency domain analysis of optical flow
- **Mode Extraction**: Identifies dominant vibration frequencies
- **Visualization**: Displays mode shapes and power spectral density

## Key Features

- **Complete from-scratch implementation** of computer vision algorithms
- **Modular design** with separate functions for each processing step
- **Real-time visualization** of optical flow and modal analysis results
- **Frequency domain analysis** to extract object vibration modes
- **Educational code structure** with clear variable names and comments

## Project Structure

```
├── main.py                    # Main execution scripts (multiple versions)
├── grayscaling.py            # RGB to grayscale conversion
├── dth_and_hysteresis.py     # Dual threshold hysteresis for edge detection
├── nonmaximumsuppression.py  # Non-maximum suppression implementation
├── ix_iy_it_calc.py         # Spatial and temporal derivative calculation
├── quiver.py                # Optical flow visualization
└── README.md                # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-space-modal-analysis
cd image-space-modal-analysis

# Install required packages
pip install numpy opencv-python matplotlib scipy
```

## Usage

### Edge Detection
```python
python main.py  # For Canny edge detection
```

### Optical Flow Analysis
```python
# Update image paths in main.py
img1path = "path/to/initial_frame.png"
img2path = "path/to/final_frame.png"
python main.py  # For Horn-Schunck optical flow
```

### Modal Analysis
```python
# Update video path in main.py
video_path = "path/to/your/video.mp4"
python main.py  # For modal analysis version
```

## Results

The implementation successfully:
- Detects edges using custom Canny implementation
- Computes optical flow between frame pairs
- Extracts vibration modes from video sequences
- Visualizes power spectral density and mode shapes
- Identifies dominant frequencies in object motion

## Example Output

- **Edge Detection**: Clean, single-pixel width edges with minimal noise
- **Optical Flow**: Vector field showing motion directions and magnitudes
- **Modal Analysis**: Frequency peaks and corresponding spatial mode shapes

## Technical Details

### Algorithms Used
1. **Sobel Operators**: For gradient calculation in both edge detection and optical flow
2. **Gaussian Filtering**: For noise reduction and motion smoothing
3. **FFT**: For frequency domain analysis of temporal motion signals
4. **Peak Detection**: For identifying dominant vibration frequencies

### Key Parameters
- **Alpha (α)**: Smoothness parameter in Horn-Schunck method (default: 25)
- **Gaussian Kernel**: 5x5 kernel with σ=3 for blur operations
- **Threshold Ratios**: Low (0.05) and high (0.09) for hysteresis
- **Iteration Count**: 50-150 iterations for optical flow convergence

## Educational Value

This project demonstrates:
- **Computer Vision Fundamentals**: Edge detection, optical flow, frequency analysis
- **Mathematical Implementation**: Converting theoretical algorithms to code
- **Signal Processing**: Time-frequency analysis using FFT
- **Visualization Techniques**: Effective display of complex motion data

## Research Connection

Based on the MIT CSAIL paper that introduces:
- **Modal Analysis Theory**: Extracting vibration modes from video
- **Image-Space Dynamics**: Analyzing object motion in 2D image coordinates
- **Frequency Domain Processing**: Using temporal spectra for mode identification

## Limitations & Future Work

### Current Limitations
- No interactive manipulation interface (from original paper)
- Limited to small motions and stable objects
- Requires manual parameter tuning for different videos
- No real-time processing capability

### Potential Extensions
- [ ] Interactive object manipulation interface
- [ ] Real-time processing optimization
- [ ] GPU acceleration using OpenCV or PyTorch
- [ ] Automatic parameter selection
- [ ] 3D mode extraction capabilities
- [ ] Integration with modern deep learning approaches

## Dependencies

```
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.5.0
scipy>=1.7.0
```

## References

1. Davis, A., Chen, J. G., & Durand, F. (2015). "Image-Space Modal Bases for Plausible Manipulation of Objects in Video." *ACM Transactions on Graphics*.

2. Horn, B. K., & Schunck, B. G. (1981). "Determining optical flow." *Artificial Intelligence*, 17(1-3), 185-203.

3. Canny, J. (1986). "A computational approach to edge detection." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679-698.

## Contributing

Feel free to contribute by:
- Adding new visualization methods
- Implementing the interactive manipulation interface
- Optimizing performance for larger videos
- Adding support for different video formats
- Improving documentation and examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIT CSAIL for the foundational research
- OpenCV community for computer vision tools
- NumPy and SciPy teams for numerical computing libraries

---

*This implementation focuses on the modal analysis aspects of the original paper, providing a solid foundation for understanding object dynamics in video through frequency domain analysis.*

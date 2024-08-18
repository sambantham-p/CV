import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
# Task 1
# recording voice
fs = 8000
dur = 5
recording = sd.rec(int(dur*fs),samplerate=fs,channels=2)
sd.wait()
wavfile.write('D:/9th sem/COMPUTER VISION/out.wav',fs,recording)

# displaying voice
fs , d = wavfile.read('D:/9th sem/COMPUTER VISION/countdown.wav')
print('fs:',fs,'d:',d)
sd.play(d,fs)
plt.plot(d)

# Task 2 - lloyd_max_quantization
class LloydMaxQuantizer:
    def __init__(self, X, R):
        self.X = X
        self.R = R
        self.n = 2**R
        self.Xmax = np.max(X)
        self.Xmin = np.min(X)
        self.DR = self.Xmax - self.Xmin
        self.T = self.Xmin + (np.arange(self.n) + 0.5) / self.n * self.DR
    
    def quantize(self, x):
        return self.T[np.argmin(np.abs(x - self.T))]
    
    def lloyd_max(self):
        converged = False
        while not converged:
            T_new = np.zeros(self.n)
            count = np.zeros(self.n)
            for x in self.X:
                idx = np.argmin(np.abs(x - self.T))
                T_new[idx] += x
                count[idx] += 1
            T_new /= count
            if np.allclose(self.T, T_new):
                converged = True
            self.T = T_new.copy()
    
    def process_and_plot(self):
        self.lloyd_max()
        y = np.array([self.quantize(x) for x in self.X])
        # Plot the results
        plt.figure(figsize=(15, 5))
        # Original signal
        plt.subplot(1, 3, 1)
        plt.plot(self.X)
        plt.title('Original Signal')
        # Quantized signal
        plt.subplot(1, 3, 2)
        plt.plot(y)
        plt.title('Quantized Signal')
        # Overlay of original and quantized signals
        plt.subplot(1, 3, 3)
        plt.plot(self.X, label='Original Signal')
        plt.plot(y, label='Quantized Signal', linestyle='--')
        plt.title('Original vs Quantized')
        plt.legend()
        plt.tight_layout()
        plt.show()
t = np.linspace(0, 10, 1000)
X = 3 * np.sin(2 * np.pi * t) + 4 * np.cos(3 * np.pi * t)
quantizer = LloydMaxQuantizer(X, R=2)
quantizer.process_and_plot()

# Task 3 - different format of img and visualize
cam = cv2.imread("cam.jpeg",0)
file_formats = ['jpg','png', 'bmp', 'tif']
file_sizes = []
for i in file_formats:
    output_path = f'cameraman.{i}'
    cv2.imwrite(output_path, cam)
    file_size = os.path.getsize(output_path)
    file_sizes.append(file_size)
    print('file format:',i,', file size is : ',file_size) 
plt.figure(figsize=(10,7))
plt.bar(file_formats, file_sizes, color='blue')
plt.xlabel('File Format')
plt.ylabel('File Size (bytes)')
plt.title('Storage Space Required for Cameraman Image in Different Formats')
plt.show()

# Task 4 - shuffle the pixels
def scramble_pixels(image):
  h,w = image.shape
  scrambled_image = image.copy()
  pixel_indices = list(range(h * w))
  random.shuffle(pixel_indices)
  for i in range(h * w):
    row_orig = i // w
    col_orig = i % w
    row_new = pixel_indices[i] // w
    col_new = pixel_indices[i] % w
    scrambled_image[row_new, col_new] = image[row_orig, col_orig]
  plt.imshow(scrambled_image, cmap='gray')
  plt.show()

scramble_pixels(cam_rgb_copy)
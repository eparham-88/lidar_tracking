import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ipywidgets import *
from scipy.ndimage import convolve1d
from scipy.signal import firwin, welch

def blur(im):
    im_blur = cv2.GaussianBlur(im, (7,7), 0)
    
    return im_blur

def remove_sides(im):
    """
    crops an image to remove the left-most and right-most portions
    
    Args:
        im (np.array): the image you want to crop

    Returns:
        cropped_im (np.array): cropped image
    """
    im_width = im.shape[1]
    left_crop = int(im_width*0.2)
    right_crop = int(im_width*0.8)
    
    cropped_im = im[:,left_crop:right_crop]
    
    return cropped_im

def remove_lines(image, distortion_freq=None, num_taps=65, eps=0.025):
    """Removes horizontal line artifacts from scanned image.
    Args:
    image: 2D or 3D array.
    distortion_freq: Float, distortion frequency in cycles/pixel, or
        `None` to estimate from spectrum.
    num_taps: Integer, number of filter taps to use in each dimension.
    eps: Small positive param to adjust filters cutoffs (cycles/pixel).
    Returns:
    Denoised image.
    """
    # image = np.asarray(image, float)
    if distortion_freq is None:
        distortion_freq = estimate_distortion_freq(image)

    hpf = firwin(num_taps, distortion_freq - eps,
                pass_zero='highpass', fs=1)
    lpf = firwin(num_taps, eps, pass_zero='lowpass', fs=1)
    return image - convolve1d(convolve1d(image, hpf, axis=0), lpf, axis=1)

def estimate_distortion_freq(image, min_frequency=1/25):
    """Estimates distortion frequency as spectral peak in vertical dim."""
    f, pxx = welch(np.reshape(image, (len(image), -1), 'C').sum(axis=1))
    pxx[f < min_frequency] = 0.0
    return f[pxx.argmax()]
    
    
def main(folder):
    wd = os.getcwd()
    
    lst = os.listdir(folder)
    lst.sort()
    
    for filename in lst:
        im = cv2.imread(os.path.join(wd,folder,filename), cv2.IMREAD_GRAYSCALE)
        im = remove_sides(im)
        
        blurred = True
        removed = False #this actually just doesn't work right now, so dont set this to true lol
        
        if blurred and removed:
            im_blurred = blur(im)
            im_removed = remove_lines(im)
            cv2.imshow("original, blurred, removed", np.vstack((im, im_blurred, im_removed)))
            pass
        elif blurred:
            im_blurred = blur(im)
            cv2.imshow("original, blurred", np.vstack((im, im_blurred)))
        elif removed:
            im_removed = remove_lines(im)
            cv2.imshow("original, removed", np.vstack((im, im_removed)))
            
        cv2.waitKey(0)
    
    pass

if __name__ == "__main__":
    main('2023_10_21_04_10_PM_lidar_camera/signal')
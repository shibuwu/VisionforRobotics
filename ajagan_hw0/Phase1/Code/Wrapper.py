#!/usr/bin/env python3

import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import convolve

def gaussian_2d(size, sigma):
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def rotate_kernel(kernel, angle):
    h, w = kernel.shape
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(kernel, rotation_matrix, (w, h), 
                              flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=0)
    return rotated

def sobel_x():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)

def create_dog_filter_bank(scales=[1, 2], orientations=16):
    filters = []
    angles = np.linspace(0, 360, orientations, endpoint=False)
    
    for scale in scales:
        size = int(6 * scale) + 1
        if size % 2 == 0:
            size += 1
        gaussian = gaussian_2d(size, scale)
        sobel = sobel_x()

        pad_size = (size - 3) // 2
        if pad_size > 0:
            sobel_padded = np.pad(sobel, pad_size, mode='constant', constant_values=0)
        else:
            sobel_padded = sobel
            
        dog_base = convolve(gaussian, sobel_padded, mode='constant')
        
        for angle in angles:
            rotated_filter = rotate_kernel(dog_base, angle)
            filters.append(rotated_filter)
    
    return filters

def visualize_filter_bank(filters, name, cols=16, save_path=None):
    n_filters = len(filters)
    rows = int(np.ceil(n_filters / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n_filters:
                axes[i, j].imshow(filters[idx], cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle(f'{name} Filter Bank ({n_filters} filters)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {name} filter bank to {save_path}")
    
    plt.close()

def gaussian_first_derivative(size, sigma_x, sigma_y, order='x'):
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)
    
    gaussian = np.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
    
    if order == 'x':
        derivative = -xx / (sigma_x**2) * gaussian
    else:
        derivative = -yy / (sigma_y**2) * gaussian
    
    derivative = derivative - np.mean(derivative)
    return derivative

def gaussian_second_derivative(size, sigma_x, sigma_y, order='xx'):
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)
    
    gaussian = np.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
    
    if order == 'xx':
        derivative = (xx**2 / sigma_x**4 - 1 / sigma_x**2) * gaussian
    elif order == 'yy':
        derivative = (yy**2 / sigma_y**4 - 1 / sigma_y**2) * gaussian
    else:
        derivative = (xx * yy) / (sigma_x**2 * sigma_y**2) * gaussian
    
    derivative = derivative - np.mean(derivative)
    return derivative

def laplacian_of_gaussian(size, sigma):
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)
    r_squared = xx**2 + yy**2
    log_kernel = -1 / (np.pi * sigma**4) * (1 - r_squared / (2 * sigma**2)) * \
                 np.exp(-r_squared / (2 * sigma**2))
    log_kernel = log_kernel - np.mean(log_kernel)
    return log_kernel

def create_lm_filter_bank(filter_type='LMS'):
    filters = []
    if filter_type == 'LMS':
        basic_scales = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]
    else:
        basic_scales = [np.sqrt(2), 2, 2 * np.sqrt(2), 4]
    
    derivative_scales = basic_scales[:3]
    orientations = 6
    angles = np.linspace(0, 180, orientations, endpoint=False)
    elongation = 3
    
    for sigma in derivative_scales:
        sigma_x = sigma
        sigma_y = elongation * sigma
        size = int(6 * sigma_y) + 1
        if size % 2 == 0:
            size += 1
        
        first_deriv = gaussian_first_derivative(size, sigma_x, sigma_y, order='x')
        second_deriv = gaussian_second_derivative(size, sigma_x, sigma_y, order='xx')
        
        for angle in angles:
            filters.append(rotate_kernel(first_deriv, angle))
        
        for angle in angles:
            filters.append(rotate_kernel(second_deriv, angle))
    
    log_scales = [basic_scales[0], 3 * basic_scales[0],
                  basic_scales[1], 3 * basic_scales[1],
                  basic_scales[2], 3 * basic_scales[2],
                  basic_scales[3], 3 * basic_scales[3]]
    
    for sigma in log_scales:
        size = int(6 * sigma) + 1
        if size % 2 == 0:
            size += 1
        filters.append(laplacian_of_gaussian(size, sigma))
    
    for sigma in basic_scales:
        size = int(6 * sigma) + 1
        if size % 2 == 0:
            size += 1
        filters.append(gaussian_2d(size, sigma))
    
    return filters

def gabor_filter(size, sigma, theta, lambd, gamma, psi=0):
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)
    
    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
    
    gb = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * \
         np.cos(2 * np.pi * x_theta / lambd + psi)
    return gb

def create_gabor_filter_bank(scales=[3, 5, 7, 9], orientations=8):
    filters = []
    angles = np.linspace(0, np.pi, orientations, endpoint=False)
    
    for sigma in scales:
        size = int(6 * sigma) + 1
        if size % 2 == 0:
            size += 1
        lambd = sigma * 2
        gamma = 0.5
        
        for theta in angles:
            gb = gabor_filter(size, sigma, theta, lambd, gamma, psi=0)
            filters.append(gb)
    
    return filters

def create_half_disk_masks(radii=[5, 10, 15], orientations=8):
    masks = []
    angles = np.linspace(0, 360, orientations, endpoint=False)
    
    for radius in radii:
        size = 2 * radius + 1
        center = radius
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        disk = dist <= radius
        
        for angle in angles:
            theta = np.radians(angle)
            x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
            side = (x_grid - center) * np.sin(theta) - (y_grid - center) * np.cos(theta)
            
            left_mask = disk & (side >= 0)
            right_mask = disk & (side < 0)
            
            left_mask = left_mask.astype(np.float64)
            right_mask = right_mask.astype(np.float64)
            
            if np.sum(left_mask) > 0:
                left_mask = left_mask / np.sum(left_mask)
            if np.sum(right_mask) > 0:
                right_mask = right_mask / np.sum(right_mask)
            
            masks.append((left_mask, right_mask))
    
    return masks

def visualize_half_disk_masks(masks, orientations, save_path=None):
    n_masks = len(masks)
    n_scales = n_masks // orientations
    
    fig, axes = plt.subplots(n_scales, orientations * 2, 
                              figsize=(orientations * 2, n_scales * 1.5))
    
    if n_scales == 1:
        axes = axes.reshape(1, -1)
    
    for scale_idx in range(n_scales):
        for orient_idx in range(orientations):
            mask_idx = scale_idx * orientations + orient_idx
            left_mask, right_mask = masks[mask_idx]
            
            ax_left = axes[scale_idx, orient_idx * 2]
            ax_left.imshow(left_mask, cmap='gray')
            ax_left.axis('off')
            
            ax_right = axes[scale_idx, orient_idx * 2 + 1]
            ax_right.imshow(right_mask, cmap='gray')
            ax_right.axis('off')
    
    plt.suptitle('Half-Disk Masks', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved half-disk masks to {save_path}")
    
    plt.close()

def apply_filter_bank(image, filters):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = image.astype(np.float64)
    
    responses = []
    for f in filters:
        response = convolve(image, f, mode='reflect')
        responses.append(response)
    
    return np.stack(responses, axis=-1)

def create_texton_map(image, filters, n_clusters=64):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    responses = apply_filter_bank(gray, filters)
    h, w, n_filters = responses.shape
    responses_flat = responses.reshape(-1, n_filters)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(responses_flat)
    texton_map = labels.reshape(h, w)
    
    return texton_map

def create_brightness_map(image, n_clusters=16):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float64)
    h, w = gray.shape
    gray_flat = gray.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(gray_flat)
    brightness_map = labels.reshape(h, w)
    
    return brightness_map

def create_color_map(image, n_clusters=16):
    if len(image.shape) == 2:
        return create_brightness_map(image, n_clusters)
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab = lab.astype(np.float64)
    h, w, c = lab.shape
    lab_flat = lab.reshape(-1, c)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(lab_flat)
    color_map = labels.reshape(h, w)
    
    return color_map

EPS = 1e-8

def compute_chi_square_gradient(map_image, masks, n_bins):
    h, w = map_image.shape
    n_masks = len(masks)
    gradients = np.zeros((h, w, n_masks))
    
    for mask_idx, (left_mask, right_mask) in enumerate(masks):
        chi_sqr_dist = np.zeros((h, w))
        
        for bin_idx in range(n_bins):
            tmp = (map_image == bin_idx).astype(np.float64)
            g_i = convolve(tmp, left_mask, mode='reflect')
            h_i = convolve(tmp, right_mask, mode='reflect')
            
            numerator = (g_i - h_i) ** 2
            denominator = g_i + h_i + EPS
            chi_sqr_dist += numerator / denominator
        
        chi_sqr_dist = chi_sqr_dist / 2
        gradients[:, :, mask_idx] = chi_sqr_dist
    
    return gradients

def compute_texture_gradient(texton_map, masks, n_clusters=64):
    return compute_chi_square_gradient(texton_map, masks, n_clusters)

def compute_brightness_gradient(brightness_map, masks, n_clusters=16):
    return compute_chi_square_gradient(brightness_map, masks, n_clusters)

def compute_color_gradient(color_map, masks, n_clusters=16):
    return compute_chi_square_gradient(color_map, masks, n_clusters)

def compute_pb_lite(Tg, Bg, Cg, canny_baseline, sobel_baseline, w1=0.5, w2=0.5):
    Tg_mean = np.mean(Tg, axis=2)
    Bg_mean = np.mean(Bg, axis=2)
    Cg_mean = np.mean(Cg, axis=2)
    
    gradient_mean = (Tg_mean + Bg_mean + Cg_mean) / 3
    gradient_mean = (gradient_mean - gradient_mean.min()) / (gradient_mean.max() - gradient_mean.min() + EPS)
    
    if len(canny_baseline.shape) == 3:
        canny_baseline = cv2.cvtColor(canny_baseline, cv2.COLOR_BGR2GRAY)
    if len(sobel_baseline.shape) == 3:
        sobel_baseline = cv2.cvtColor(sobel_baseline, cv2.COLOR_BGR2GRAY)
    
    canny_baseline = canny_baseline.astype(np.float64) / 255.0
    sobel_baseline = sobel_baseline.astype(np.float64) / 255.0
    
    baseline_combined = w1 * canny_baseline + w2 * sobel_baseline
    pb_lite = gradient_mean * baseline_combined
    pb_lite = (pb_lite - pb_lite.min()) / (pb_lite.max() - pb_lite.min() + EPS)
    
    return pb_lite

def save_map_visualization(map_image, save_path, title="Map"):
    plt.figure(figsize=(8, 8))
    plt.imshow(map_image, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to {save_path}")

def save_gradient_visualization(gradient, save_path, title="Gradient"):
    gradient_mean = np.mean(gradient, axis=2)
    plt.figure(figsize=(8, 8))
    plt.imshow(gradient_mean, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to {save_path}")

def save_pb_lite_visualization(pb_lite, save_path, title="Pb-lite Output"):
    plt.figure(figsize=(8, 8))
    plt.imshow(pb_lite, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to {save_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "..", "BSDS500")
    images_path = os.path.join(base_path, "Images")
    canny_path = os.path.join(base_path, "CannyBaseline")
    sobel_path = os.path.join(base_path, "SobelBaseline")
    output_path = os.path.join(script_dir, "..", "Output")
    
    os.makedirs(output_path, exist_ok=True)
    
    print("PB-LITE BOUNDARY DETECTION")
    print("-" * 40)
    
    print("\nGenerating Filter Banks...")
    dog_filters = create_dog_filter_bank(scales=[1, 2], orientations=16)
    visualize_filter_bank(dog_filters, "DoG", cols=16, 
                          save_path=os.path.join(output_path, "DoG.png"))
    
    lms_filters = create_lm_filter_bank(filter_type='LMS')
    visualize_filter_bank(lms_filters, "LM_Small", cols=12, 
                          save_path=os.path.join(output_path, "LMS.png"))
    
    lml_filters = create_lm_filter_bank(filter_type='LML')
    visualize_filter_bank(lml_filters, "LM_Large", cols=12, 
                          save_path=os.path.join(output_path, "LML.png"))
    
    gabor_filters = create_gabor_filter_bank(scales=[3, 5, 7, 9], orientations=8)
    visualize_filter_bank(gabor_filters, "Gabor", cols=8, 
                          save_path=os.path.join(output_path, "Gabor.png"))
    
    all_filters = dog_filters + lms_filters + lml_filters + gabor_filters
    print(f"Total filters: {len(all_filters)}")
    
    print("\nGenerating Half-Disk Masks...")
    half_disk_masks = create_half_disk_masks(radii=[5, 10, 15], orientations=8)
    visualize_half_disk_masks(half_disk_masks, orientations=8, 
                               save_path=os.path.join(output_path, "HDMasks.png"))
    print(f"Total mask pairs: {len(half_disk_masks)}")
    
    print("\nProcessing Images...")
    
    if os.path.exists(images_path):
        image_files = [f for f in os.listdir(images_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    else:
        print(f"Warning: Images path not found: {images_path}")
        print("Creating sample output with a test image...")
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_files = []
    
    for img_file in image_files:
        print(f"\nProcessing: {img_file}")
        img_name = os.path.splitext(img_file)[0]
        
        img_output_path = os.path.join(output_path, img_name)
        os.makedirs(img_output_path, exist_ok=True)
        
        image = cv2.imread(os.path.join(images_path, img_file))
        if image is None:
            print(f"Error reading {img_file}, skipping...")
            continue
        
        print("  Creating Texton Map...")
        texton_map = create_texton_map(image, all_filters, n_clusters=64)
        save_map_visualization(texton_map, 
                               os.path.join(img_output_path, "TextonMap.png"),
                               f"Texton Map - {img_name}")
        
        print("  Creating Brightness Map...")
        brightness_map = create_brightness_map(image, n_clusters=16)
        save_map_visualization(brightness_map, 
                               os.path.join(img_output_path, "BrightnessMap.png"),
                               f"Brightness Map - {img_name}")
        
        print("  Creating Color Map...")
        color_map = create_color_map(image, n_clusters=16)
        save_map_visualization(color_map, 
                               os.path.join(img_output_path, "ColorMap.png"),
                               f"Color Map - {img_name}")
        
        print("  Computing Texture Gradient...")
        Tg = compute_texture_gradient(texton_map, half_disk_masks, n_clusters=64)
        save_gradient_visualization(Tg, 
                                    os.path.join(img_output_path, "Tg.png"),
                                    f"Texture Gradient - {img_name}")
        
        print("  Computing Brightness Gradient...")
        Bg = compute_brightness_gradient(brightness_map, half_disk_masks, n_clusters=16)
        save_gradient_visualization(Bg, 
                                    os.path.join(img_output_path, "Bg.png"),
                                    f"Brightness Gradient - {img_name}")
        
        print("  Computing Color Gradient...")
        Cg = compute_color_gradient(color_map, half_disk_masks, n_clusters=16)
        save_gradient_visualization(Cg, 
                                    os.path.join(img_output_path, "Cg.png"),
                                    f"Color Gradient - {img_name}")
        
        canny_file = os.path.join(canny_path, f"{img_name}.png")
        sobel_file = os.path.join(sobel_path, f"{img_name}.png")
        
        canny_baseline = cv2.imread(canny_file)
        sobel_baseline = cv2.imread(sobel_file)
        
        print("  Computing Pb-lite output...")
        pb_lite = compute_pb_lite(Tg, Bg, Cg, canny_baseline, sobel_baseline, 
                                   w1=0.5, w2=0.5)
        save_pb_lite_visualization(pb_lite, 
                                    os.path.join(img_output_path, "PbLite.png"),
                                    f"Pb-lite Output - {img_name}")
    
    print("\n" + "-" * 40)
    print("Processing complete.")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    main()
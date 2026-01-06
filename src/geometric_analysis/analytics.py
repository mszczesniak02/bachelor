
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, closing
from skimage import data, morphology, measure
from skimage.util import invert
import cv2

import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

import os
from PIL import Image


def preprocess_mask(mask_input, threshold: float = 0.5, min_size: int = 50, hole_threshold: int = 30) -> np.ndarray:

    if isinstance(mask_input, str):
        if not os.path.exists(mask_input):
            raise FileNotFoundError(f"Mask file not found: {mask_input}")
        mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            try:
                mask = np.array(Image.open(mask_input).convert('L'))
            except Exception:
                raise ValueError(f"Could not load mask from {mask_input}")
    elif torch.is_tensor(mask_input):
        mask = mask_input.cpu().numpy()
    else:
        mask = mask_input

    if mask.max() > 1:
        mask = mask / 255.0

    binary_mask = mask > threshold

    mask_clean = remove_small_objects(binary_mask, min_size=min_size)

    mask_filled = remove_small_holes(mask_clean, area_threshold=hole_threshold)
    mask_closed = closing(mask_filled, footprint=np.ones((3, 3)))

    return mask_closed.astype(bool)


def calculate_basic_properties(binary_mask: np.ndarray):
    label_img = measure.label(binary_mask)
    regions = measure.regionprops(label_img)

    results = {
        "area_pixels": 0,
        "perimeter_pixels": 0,
        "bbox": None,
        "orientation": 0,
        "centroid": (0, 0),
        "eccentricity": 0,
        "axis_major_length": 0,
        "axis_minor_length": 0,
        "aspect_ratio": 0,
        "feret_diameter_max": 0,
        "solidity": 0,
        "extent": 0
    }

    if not regions:
        return results

    # Assuming the crack is the largest region for shape descriptors
    largest_region = max(regions, key=lambda r: r.area)

    results["area_pixels"] = sum(r.area for r in regions)
    results["perimeter_pixels"] = sum(r.perimeter for r in regions)

    # Properties of largest segment
    results["bbox"] = largest_region.bbox
    results["orientation"] = largest_region.orientation
    results["centroid"] = largest_region.centroid
    results["eccentricity"] = largest_region.eccentricity
    results["axis_major_length"] = largest_region.axis_major_length
    results["axis_minor_length"] = largest_region.axis_minor_length

    if largest_region.axis_minor_length > 0:
        results["aspect_ratio"] = largest_region.axis_major_length / \
            largest_region.axis_minor_length
    else:
        results["aspect_ratio"] = 0

    try:
        results["feret_diameter_max"] = largest_region.feret_diameter_max
    except AttributeError:
        # Fallback for older skimage versions
        results["feret_diameter_max"] = 0

    results["solidity"] = largest_region.solidity
    results["extent"] = largest_region.extent

    return results


def get_skeleton(binary_mask: np.ndarray) -> np.ndarray:

    return skeletonize(binary_mask)


def calculate_length(skeleton: np.ndarray) -> float:

    return np.sum(skeleton)


def calculate_width(binary_mask: np.ndarray, skeleton: np.ndarray):

    dist_transform = ndimage.distance_transform_edt(binary_mask)
    skeleton_coords = np.argwhere(skeleton)  # [N, 2] -> (y, x)
    if len(skeleton_coords) == 0:
        return {
            "mean_width": 0.0,
            "max_width": 0.0,
            "min_width": 0.0,
            "std_width": 0.0,
            "widths": [],
            "max_width_loc": None,
            "min_width_loc": None
        }, dist_transform

    skeleton_width_values = dist_transform[skeleton_coords[:,
                                                           0], skeleton_coords[:, 1]] * 2

    # Find locations of max and min
    idx_max = np.argmax(skeleton_width_values)
    idx_min = np.argmin(skeleton_width_values)

    max_loc = skeleton_coords[idx_max]  # [y, x]
    min_loc = skeleton_coords[idx_min]  # [y, x]

    stats = {
        "mean_width": np.mean(skeleton_width_values),
        "max_width": np.max(skeleton_width_values),
        "min_width": np.min(skeleton_width_values),
        "std_width": np.std(skeleton_width_values),
        "widths": skeleton_width_values,
        # (y, x) for consistency
        "max_width_loc": (int(max_loc[0]), int(max_loc[1])),
        "min_width_loc": (int(min_loc[0]), int(min_loc[1]))
    }

    return stats, dist_transform


def calculate_advanced_metrics(skeleton: np.ndarray, binary_mask: np.ndarray):

    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

    neighbors = convolve(skeleton.astype(
        int), neighbor_kernel, mode='constant', cval=0)
    skeleton_neighbors = neighbors * skeleton

    endpoints = np.sum(skeleton_neighbors == 1)
    branch_points = np.sum(skeleton_neighbors > 2)

    coords = np.column_stack(np.where(skeleton))
    if len(coords) > 1:
        dist_euclid = np.linalg.norm(coords[0] - coords[-1])
        skeleton_length = np.sum(skeleton)
        tortuosity = skeleton_length / dist_euclid if dist_euclid > 0 else 1.0
    else:
        tortuosity = 1.0

    def box_count(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
            np.arange(0, img.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])

    pixels = np.array(binary_mask)
    if np.sum(pixels) > 0:
        p = min(pixels.shape)
        n = 2**np.floor(np.log2(p))
        pixels_cut = pixels[0:int(n), 0:int(n)]

        scales = np.logspace(1, int(np.log2(n)), num=10,
                             endpoint=False, base=2).astype(int)
        # remove duplicate scales
        scales = np.unique(scales)
        # remove scales larger than image
        scales = scales[scales < n]
        if len(scales) == 0:
            scales = [1]

        counts = [box_count(pixels_cut, s) for s in scales]

        # Fit line
        if len(scales) > 1:
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            fractal_dim = -coeffs[0]
        else:
            fractal_dim = 0
    else:
        fractal_dim = 0

    # 4. Additional Metrics
    skeleton_length = np.sum(skeleton)
    image_area = binary_mask.size

    branching_intensity = branch_points / \
        skeleton_length if skeleton_length > 0 else 0.0
    crack_density = skeleton_length / image_area if image_area > 0 else 0.0

    return {
        "endpoints_count": int(endpoints),
        "branch_points_count": int(branch_points),
        "tortuosity": float(tortuosity),
        "fractal_dimension": float(fractal_dim),
        "branching_intensity": float(branching_intensity),
        "crack_density": float(crack_density)
    }


def analyze_crack_mask(mask: np.ndarray, pixel_size_mm: float = None):
    """
    Main analysis function.
    pixel_size_mm: if provided, converts pixels to physical units (mm).
    """
    binary_mask = preprocess_mask(mask)

    # 1. Basic Props
    basic_props = calculate_basic_properties(binary_mask)

    # 2. Skeleton & Length
    skeleton = get_skeleton(binary_mask)
    length_pixels = calculate_length(skeleton)

    # 3. Width
    width_stats, dist_map = calculate_width(binary_mask, skeleton)

    # 4. Advanced Metrics
    advanced_props = calculate_advanced_metrics(skeleton, binary_mask)

    analysis_results = {
        "basic": basic_props,
        "advanced": advanced_props,
        "length_pixels": length_pixels,
        "width_stats": width_stats
    }

    # Conversion if pixel size known
    if pixel_size_mm:
        analysis_results["length_mm"] = length_pixels * pixel_size_mm
        analysis_results["width_mean_mm"] = width_stats["mean_width"] * pixel_size_mm
        analysis_results["width_max_mm"] = width_stats["max_width"] * \
            pixel_size_mm
        analysis_results["area_mm2"] = basic_props["area_pixels"] * \
            (pixel_size_mm ** 2)

    return analysis_results, skeleton, dist_map


def visualize_analysis(image, mask: np.ndarray, skeleton: np.ndarray, dist_map: np.ndarray, results: dict, save_path: str = None):
    """
    Plots the analysis results: Original, Skeleton on Mask, Width/Distance Map.
    Accepts image as numpy array or file path (optional).
    """
    if isinstance(image, str) and os.path.exists(image):
        image = cv2.imread(image)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4 if image is not None else 3, figsize=(
        20 if image is not None else 15, 5))

    idx = 0
    if image is not None:
        axes[idx].imshow(image)
        axes[idx].set_title("Original Image")
        axes[idx].axis('off')
        idx += 1

    # Original Image with Mask overlay (optional) or just Mask
    axes[idx].imshow(mask, cmap='gray')
    axes[idx].set_title("Binary Mask (Processed)")
    axes[idx].axis('off')
    idx += 1

    # Skeleton
    axes[idx].imshow(mask, cmap='gray', alpha=0.3)
    axes[idx].imshow(skeleton, cmap='jet', alpha=0.7)
    axes[idx].set_title(f"Skeleton (Len: {results['length_pixels']:.1f} px)")
    axes[idx].axis('off')
    idx += 1

    # Width Map
    # Mask dist map by binary_mask to clear background
    masked_dist_map = dist_map * mask
    im3 = axes[idx].imshow(masked_dist_map, cmap='magma')
    axes[idx].set_title(
        f"Width Map (Max: {results['width_stats']['max_width']:.1f} px)")
    axes[idx].axis('off')
    plt.colorbar(im3, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Analysis visualization saved to {save_path}")


if __name__ == "__main__":
    # Test with provided image
    image_path = "/home/krzeslaav/Projects/bachlor/src/final_prediction_pipeline/output_predictions/image_test_0_ensemble_binary.png"
    print(f"Running analytics test on {image_path}...")

    if os.path.exists(image_path):
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, mask_thresh = cv2.threshold(
            img_gray, 80, 255, cv2.THRESH_BINARY_INV)
        mask_thresh = mask_thresh > 0  # make boolean

        results, skel, dmap = analyze_crack_mask(
            mask_thresh, pixel_size_mm=0.26)  # assumption: 0.26mm/px

        print("Analysis Results (Pixels):")
        print(
            f"  [Basic] Area: {results['basic']['area_pixels']} px ({results.get('area_mm2', 0):.2f} mm2)")
        print(
            f"  [Basic] Perimeter: {results['basic']['perimeter_pixels']:.1f} px")
        print(f"  [Basic] Solidity: {results['basic']['solidity']:.3f}")
        print(
            f"  [Basic] Aspect Ratio: {results['basic']['aspect_ratio']:.2f}")
        print(
            f"  [Basic] Orientation: {results['basic']['orientation']:.2f} rad ({(results['basic']['orientation']*180/np.pi):.1f} deg)")
        print(
            f"  [Basic] Eccentricity: {results['basic']['eccentricity']:.3f}")
        print(f"  [Basic] Extent: {results['basic']['extent']:.3f}")
        print(
            f"  [Basic] Feret Max: {results['basic']['feret_diameter_max']:.1f} px")

        print(
            f"  [Width] Length: {results['length_pixels']:.1f} px ({results.get('length_mm', 0):.2f} mm)")
        print(
            f"  [Width] Mean Width: {results['width_stats']['mean_width']:.2f} px ({results.get('width_mean_mm', 0):.2f} mm)")
        print(
            f"  [Width] Max Width: {results['width_stats']['max_width']:.2f} px ({results.get('width_max_mm', 0):.2f} mm)")
        print(
            f"  [Width] Min Width: {results['width_stats']['min_width']:.2f} px")
        print(
            f"  [Width] Std Width: {results['width_stats']['std_width']:.2f} px")

        if 'advanced' in results:
            adv = results['advanced']
            print(f"  [Advanced] Tortuosity: {adv['tortuosity']:.3f}")
            print(f"  [Advanced] Fractal Dim: {adv['fractal_dimension']:.3f}")
            print(f"  [Advanced] Branch Points: {adv['branch_points_count']}")
            print(f"  [Advanced] Endpoints: {adv['endpoints_count']}")
            print(
                f"  [Advanced] Branching Intensity: {adv['branching_intensity']:.4f}")
            print(f"  [Advanced] Crack Density: {adv['crack_density']:.4f}")

        visualize_analysis(img_rgb, mask_thresh, skel, dmap,
                           results, save_path="analytics_output.png")
    else:
        print("Test image not found, using dummy data.")
        fake_mask = np.zeros((100, 100))
        fake_mask[20:80, 45:55] = 1
        fake_mask[50:60, 20:80] = 1

        results, skel, dmap = analyze_crack_mask(fake_mask)
        print("Results:", results)
        visualize_analysis(None, fake_mask, skel, dmap, results)

#autopep8:off
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import sys
import math

# Ensure src is in python path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))

from src.classification.common import hparams

def visualize_samples(output_path="classification_samples.png"):
    train_dir = hparams.TRAIN_DIR

    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return

    classes = sorted([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))])

    if not classes:
        print("No classes found.")
        return

    num_classes = len(classes)
    cols_per_class = 2
    categories_per_row = 2
    total_cols = categories_per_row * cols_per_class  # 4 cols
    total_rows = math.ceil(num_classes / categories_per_row)

    fig, axes = plt.subplots(total_rows, total_cols,
                             figsize=(total_cols * 3, total_rows * 4))
    if total_rows == 1:
        axes = axes.reshape(1, -1)



    for i, class_name in enumerate(classes):
        class_dir = os.path.join(train_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        selected_images = random.sample(images, min(len(images), 2))
        row_idx = i // categories_per_row
        col_start_idx = (i % categories_per_row) * cols_per_class

        title_ax = axes[row_idx, col_start_idx]
        title_ax.set_title(class_name, fontsize=14,
                           fontweight='bold', loc='left')

        for j in range(cols_per_class):
            current_col = col_start_idx + j
            ax = axes[row_idx, current_col]

            if j < len(selected_images):
                img_path = os.path.join(class_dir, selected_images[j])
                try:
                    # Open and convert to grayscale ('L')
                    img = Image.open(img_path).convert('L')
                    ax.imshow(img, cmap='gray')
                    ax.axis("off")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    ax.axis("off")
            else:
                ax.axis("off")

    total_subplots = total_rows * total_cols

    flat_axes = axes.flatten()
    used_subplots_count = num_classes * cols_per_class

    for k in range(used_subplots_count, len(flat_axes)):
        flat_axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    visualize_samples()

from torch import cuda
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

SEGFORMER_PATH = os.path.join(ROOT, "models/segformer.pth")
UNET_PATH = os.path.join(ROOT, "models/unet.pth")
YOLO_PATH = os.path.join(ROOT, "models/yolo.pt")
EFFICIENTNET_PATH = os.path.join(
    ROOT, "models/efficientnet.pth")
CONVNEXT_PATH = os.path.join(ROOT, "models/convnext.pth")
DOMAIN_CONTROLLER_PATH = os.path.join(
    ROOT, "models/domain_controller.pth")

# --- Config ---
OUTPUT_IMAGE_SIZE = 512
DEVICE = "cpu" if cuda.is_available() else "cpu"
NUM_CLASSES = 4
IMAGE_PATH_0 = os.path.join(ROOT, "image_test_0.jpg")
IMAGE_PATH_1 = os.path.join(ROOT, "image_test_1.jpg")

# --- Dataset ---
SEG_IMG_TEST_PATH = os.path.join(
    ROOT, "datasets/dataset_segmentation/test_img")
SEG_MASK_TEST_PATH = os.path.join(
    ROOT, "datasets/dataset_segmentation/test_lab")

CLASS_IMG_TEST_PATH_ROOT = os.path.join(
    ROOT, "datasets/classification_width/test_img")

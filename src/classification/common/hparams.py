import os

ON_COLAB = False

try:
    import google.colab
    DATA_DIR = "/content/datasets"
    TRAIN_DIR = os.path.join(DATA_DIR, "classification_width/train_img")
    TEST_DIR = os.path.join(DATA_DIR, "classification_width/test_img")
    DEVICE = "cuda"
    WORKERS = 0
    ON_COLAB = True

    ENET_MODEL_TRAIN_DIR = "/content/models/classification/efficienet/"
    ENET_MODEL_TRAIN_LOG_DIR = "/content/models_log/classification/efficienet/"

    CONVNEXT_MODEL_TRAIN_DIR = "/content/models/classification/convnext/"
    CONVNEXT_MODEL_TRAIN_LOG_DIR = "/content/models_log/classification/convnext/"

except ImportError:
    # using local 2GB laptop :|
    PROJECT_ROOT = os.path.abspath(os.path.join(
        # Adjusted to match the new classification paths' relative root
        os.path.dirname(__file__), "../../"))

    # Segmentation paths
    MASK_TRAIN_PATH = os.path.join(
        PROJECT_ROOT, "datasets/dataset_segmentation/train_lab")
    IMG_TRAIN_PATH = os.path.join(
        PROJECT_ROOT, "datasets/dataset_segmentation/train_img")
    MASK_TEST_PATH = os.path.join(
        PROJECT_ROOT, "datasets/dataset_segmentation/test_lab")
    IMG_TEST_PATH = os.path.join(
        PROJECT_ROOT, "datasets/dataset_segmentation/test_img")

    YOLO_DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets/yolo_seg_data")

    UNET_MODEL_TRAIN_DIR = os.path.join(PROJECT_ROOT, "models/")
    UNET_MODEL_TRAIN_LOG_DIR = os.path.join(
        # Adjusted to use PROJECT_ROOT
        PROJECT_ROOT, "assets/training_logs/segmentation/unet/")

    SEGFORMER_MODEL_TRAIN_DIR = os.path.join(PROJECT_ROOT, "models/")
    SEGFORMER_MODEL_TRAIN_LOG_DIR = os.path.join(
        # Adjusted to use PROJECT_ROOT
        PROJECT_ROOT, "assets/training_logs/segmentation/segformer/")

    # Classification paths (updated)
    TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets/entry_dataset/train/")
    TEST_DIR = os.path.join(PROJECT_ROOT, "datasets/entry_dataset/test/")
    VAL_DIR = os.path.join(PROJECT_ROOT, "datasets/entry_dataset/val/")

    MODEL_DIR = os.path.join(PROJECT_ROOT, "models/")
    # Adjusted to use PROJECT_ROOT
    LOG_DIR = os.path.join(
        PROJECT_ROOT, "assets/training_logs/entry_classificator/")

    # Updated to use MODEL_DIR or PROJECT_ROOT
    ENET_MODEL_TRAIN_DIR = os.path.join(
        PROJECT_ROOT, "models/classification/efficienet/")
    ENET_MODEL_TRAIN_LOG_DIR = os.path.join(
        # Adjusted to use PROJECT_ROOT
        PROJECT_ROOT, "assets/training_logs/classification/efficienet/")

    # Updated to use MODEL_DIR or PROJECT_ROOT
    CONVNEXT_MODEL_TRAIN_DIR = os.path.join(
        PROJECT_ROOT, "models/classification/convnext/")
    CONVNEXT_MODEL_TRAIN_LOG_DIR = os.path.join(
        # Adjusted to use PROJECT_ROOT
        PROJECT_ROOT, "assets/training_logs/classification/convnext/")

    DEVICE = "cuda"
    WORKERS = 4
    ON_COLAB = False

    # Inference paths (adjusted for PROJECT_ROOT and corrected syntax)
    MODEL_INFERENCE_DIR = os.path.join(PROJECT_ROOT, "models/")
    MODEL_INFERENCE_PATH = os.path.join(MODEL_INFERENCE_DIR, "segformer.pth")


# Common
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EPOCHS = 50
PATIENCE = 10
NUM_CLASSES = 4
SEED = 42

# EfficientNet
ENET_IMAGE_SIZE = 256
ENET_BATCH_SIZE = 16
ENET_LEARNING_RATE = 2e-5
ENET_WEIGHT_DECAY = 1e-4
ENET_EPOCHS = 15
ENET_SCHEDULER_PATIENCE = 5

# ConvNext
CONVNEXT_IMAGE_SIZE = 256
CONVNEXT_BATCH_SIZE = 16
CONVNEXT_LEARNING_RATE = 2e-5
CONVNEXT_WEIGHT_DECAY = 1e-4
CONVNEXT_EPOCHS = 15
CONVNEXT_SCHEDULER_PATIENCE = 5
MODEL_PATH = ENET_MODEL_TRAIN_DIR + "best_model.pth"
IMAGE_SIZE = 256


def main():
    print("nothing to do")


if __name__ == "__main__":
    main()

# autopep8: off
import sys
import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

original_sys_path = sys.path.copy()
# moving to "segmentation/"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))
# importing commons
from segmentation.common.dataloader import *
from segmentation.common.hparams import *
# importing utils
from utils.utils import *
# go back to the origin path
sys.path = original_sys_path
# normal imports

MODEL_PATH = os.path.join(PROJECT_ROOT, "models/yolo.pt")

def yolo_predict_single(model, dataset, index=0):
    t_img, t_msk = dataset[index]
    img_numpy = ten2np(t_img, denormalize=True)
    mask_gt = ten2np(t_msk) # Ground truth mask
    results = model.predict(img_numpy, imgsz=512, verbose=False)
    mask_pred = np.zeros((img_numpy.shape[0], img_numpy.shape[1]), dtype=np.float32)

    if results and results[0].masks is not None:
        data = results[0].masks.data


        if data.shape[1:] != (img_numpy.shape[0], img_numpy.shape[1]):
            data = data.float()
            data = F.interpolate(data.unsqueeze(1), size=(img_numpy.shape[0], img_numpy.shape[1]),
                                 mode='bilinear', align_corners=False).squeeze(1)


        mask_pred_tensor = torch.any(data > 0.5, dim=0).float()
        mask_pred = mask_pred_tensor.cpu().numpy()

    return img_numpy, mask_gt, mask_pred


def main():
    # Helper: Check paths
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    dataset = dataset_get(img_path=IMG_TEST_PATH,
                          mask_path=MASK_TEST_PATH, transform=val_transform)

    magic = 4
    img, msk, out = yolo_predict_single(model, dataset, magic)
    plot_effect(img, msk, effect=out, effect_title="Wyjście modelu")


if __name__ == "__main__":
    main()

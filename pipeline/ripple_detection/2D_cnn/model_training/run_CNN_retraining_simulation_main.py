from pathlib import Path

from demo_training_functions_m import *
from torchvision import models

# %%
# set the path to images, and architecture.
"""
1. image_path is a directory of spectrogram images (jpg) containing subfolders `train`, `test`, and `valid`.
2. train and valid have subfolders `Yes` and `No` of positive and negative image cases, respectively.
3. test has uncategorized test images.
Check to confirm more than 1 image in each subfolder.
"""

image_path = Path("training_data/spectral_image")
check_image_count(image_path)
data = get_the_data(image_path, models.resnet34)

# fine-tuning the old model
old_weight_path = Path("full_trained_model.pkl")
code_dir = Path.cwd()
filename = code_dir / "retraining_saved_model.pth"
model = train_the_model(
    data, models.resnet34, filename, old_weight_path=old_weight_path
)

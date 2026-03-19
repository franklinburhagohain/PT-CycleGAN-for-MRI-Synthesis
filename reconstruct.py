import os
import time
import gc
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from Utils import load_checkpoint
from Dataset2 import ABDataset
from Model import Generator

gc.collect()
torch.cuda.empty_cache()

input_dir = "3T/test"  # TODO: replace with your input dataset path
output_dir = "fake_dataset_output"  # TODO: replace with your desired output folder
checkpoint = "path/to/your_checkpoint.pth"  # TODO: replace with your generator checkpoint
image_width = 256
image_height = 256
batch_size = 1


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir_exists(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def masking(a, b):
    l_top = l_bottom = 0
    a = a[0]
    b = b[0]

    for i in range(a.shape[1]):
        if torch.sum(a[:, i, :]) != 0:
            break
        l_top += 1

    for i in range(a.shape[1]):
        if torch.sum(a[:, a.shape[1] - i - 1, :]) != 0:
            break
        l_bottom += 1

    b[:, :l_top, :] = 0
    b[:, b.shape[1] - l_bottom:, :] = 0

    return a, b

NUM_HEADS = [8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32] 
current_incremental_layer_index = 12
patch_size = 8
gen = Generator(width=image_width, height=image_height,att_heads=NUM_HEADS).to(device)
gen.current_incremental_layer_index = current_incremental_layer_index
load_checkpoint(checkpoint, gen, None, None)
gen.eval()
transforms = A.Compose(
    [
        A.Resize(width=image_width, height=image_height),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
) 
val_dataset = ABDataset(root_a=input_dir, transform=transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
loop = tqdm(val_loader, leave=True)

start = time.time()

for idx, (image, filename) in enumerate(loop):
    image = image.to(device)

    with torch.amp.autocast('cuda'):
        gen_image = gen(image, patch_size)
        image, gen_image = masking(image*0.5+0.5, gen_image*0.5+0.5)
 
        for img, fname in zip(gen_image, filename):
            save_path = os.path.join(output_dir, fname)
            save_image(img, save_path)

end = time.time()

print(f"Processing completed in {round(end - start, 3)} seconds.")

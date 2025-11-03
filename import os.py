import os
from PIL import Image
import cv2
import shutil
import numpy as np
from tqdm import tqdm
# Paths
SOURCE_ROOT = r"C:/Users/ishit/OneDrive/Desktop/projects/cv project/cv_project_data"
DEST_ROOT   = r"C:/Users/ishit/OneDrive/Desktop/projects/cv project/cv_project_data_preprocessed"
SIZE = (640, 640)
def preprocess_image(img_path, save_path, size=SIZE):
    # Load image
    img = Image.open(img_path)
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Convert to numpy array for OpenCV processing
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # Contrast enhancement: CLAHE on each channel
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
    # Make PIL image for resizing
    img = Image.fromarray(img_enhanced)
    # Only downscale; do not upscale small images
    if img.size[0] > size[0] or img.size[1] > size[1]:
        img = letterbox_image(img, size)
    # Optionally, for small images, just center them in a 640x640 blank (letterbox)
    elif img.size != size:
        img = letterbox_image(img, size)
    img.save(save_path)
    
def letterbox_image(image, target_size):
    # Maintains aspect ratio; pads with gray
    iw, ih = image.size
    w, h = target_size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.LANCZOS)
    new_image = Image.new('RGB', (w, h), (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def clean_and_preprocess(image_folder, label_folder, dest_image_folder, dest_label_folder):
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_label_folder, exist_ok=True)
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = 0
    for img_name in tqdm(images):
        base = os.path.splitext(img_name)[0]
        label_file = os.path.join(label_folder, base + '.txt')
        if not os.path.exists(label_file):
            continue  # skip images without matching label
        dest_img = os.path.join(dest_image_folder, img_name)
        dest_lbl = os.path.join(dest_label_folder, base + '.txt')
        try:
            preprocess_image(os.path.join(image_folder, img_name), dest_img)
            shutil.copy(label_file, dest_lbl)
            count += 1
        except Exception as ex:
            print(f"Error processing {img_name}: {ex}")
    print(f"Total processed {count} images for {dest_image_folder}")

if __name__ == "__main__":
    for split in ['train', 'val']:
        img_dir = os.path.join(SOURCE_ROOT, 'images', split)
        lbl_dir = os.path.join(SOURCE_ROOT, 'labels', split)
        dest_img_dir = os.path.join(DEST_ROOT, 'images', split)
        dest_lbl_dir = os.path.join(DEST_ROOT, 'labels', split)
        clean_and_preprocess(img_dir, lbl_dir, dest_img_dir, dest_lbl_dir)


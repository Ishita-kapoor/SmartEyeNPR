from ultralytics import YOLO

# Load pre-trained yolov8n model (nano, lightweight)
model = YOLO('yolov8n.pt')

# Train on your dataset
model.train(
    data='C:/Users/ishit/OneDrive/Desktop/projects/cv project/number_plate.yaml',
    epochs=15,
    imgsz=416,
    batch=8,
    name='plate_detection_augmented',
    augment=True,            # Enable default augmentations
    auto_augment='randaugment',  # Optional, use advanced augment policy
    mosaic=1.0,              # Mosaic intensity (default 1.0)
    mixup=0.5,               # Mixup intensity (0 to 1)
    fliplr=0.5,              # Horizontal flip probability
    flipud=0.0,              # Vertical flip probability
    hsv_h=0.015,             # HSV hue augmentation range
    hsv_s=0.7,               # HSV saturation augmentation range
    hsv_v=0.4                # HSV value augmentation range
)

# After training, your best weights are in:
# runs/detect/plate_detection/weights/best.pt

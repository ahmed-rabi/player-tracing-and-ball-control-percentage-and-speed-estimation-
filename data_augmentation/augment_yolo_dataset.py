import os
import cv2
import albumentations as A

# Input paths
image_dir = 'D:\\football_analysis\\training\\football-players-detection-1\\football-players-detection-1\\train\\images'
label_dir = 'D:\\football_analysis\\training\\football-players-detection-1\\football-players-detection-1\\train\\labels'

# Output paths
output_image_dir = 'D:\\football_analysis\\training\\football-players-detection-1\\football-players-detection-1\\train\\images_aug'
output_label_dir = 'D:\\football_analysis\\training\\football-players-detection-1\\football-players-detection-1\\train\\labels_aug'


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Loop through each image

for file in os.listdir(image_dir):
    if not file.endswith(".jpg") and not file.endswith(".png"):
        continue

    img_path = os.path.join(image_dir, file)
    label_path = os.path.join(label_dir, file.replace(".jpg", ".txt").replace(".png", ".txt"))

    # Load image and label
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        bbox = list(map(float, parts[1:]))
        bboxes.append(bbox)
        class_labels.append(class_id)

    # Apply transform
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

    aug_img = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    # Save augmented image and label
    aug_filename = file.replace(".jpg", "_aug.jpg").replace(".png", "_aug.png")
    cv2.imwrite(os.path.join(output_image_dir, aug_filename), aug_img)

    with open(os.path.join(output_label_dir, aug_filename.replace(".jpg", ".txt").replace(".png", ".txt")), 'w') as f:
        for bbox, class_id in zip(aug_bboxes, aug_labels):
            f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
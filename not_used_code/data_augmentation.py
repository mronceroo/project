import cv2
import os
import numpy as np
from albumentations import (   # Instala con: pip install albumentations
    Compose, Rotate, RandomBrightnessContrast, 
    HorizontalFlip, GaussianBlur, ShiftScaleRotate
)
from tqdm import tqdm  # Barra de progreso

# Configuración
INPUT_DIR = "dataset/training_dataset"
OUTPUT_DIR = "dataset/aumentado2"
AUGMENTATIONS_PER_IMAGE = 100  # Número de variaciones por imagen

# Transformaciones (ajusta parámetros según necesidad)
augmenter = Compose([
    Rotate(limit=15, p=0.7),  # Rotación hasta ±15 grados
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    GaussianBlur(blur_limit=(1, 3), p=0.3),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5),  # Pequeños desplazamientos
])

# Aumenta todas las imágenes
for class_dir in tqdm(os.listdir(INPUT_DIR)):
    class_path = os.path.join(INPUT_DIR, class_dir)
    output_class_path = os.path.join(OUTPUT_DIR, class_dir)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Guarda la imagen original (opcional)
        cv2.imwrite(os.path.join(output_class_path, f"orig_{img_name}"), img)

        # Genera variaciones aumentadas
        for i in range(AUGMENTATIONS_PER_IMAGE):
            augmented = augmenter(image=img)["image"]
            cv2.imwrite(
                os.path.join(output_class_path, f"aug_{i}_{img_name}"), 
                augmented
            )

print(f"¡Dataset aumentado guardado en {OUTPUT_DIR}!")
import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Configuración
OUTPUT_DIR = "dataset/colores_sinteticos"
IMAGE_SIZE = (100, 100)  # Mismo tamaño que en tu CNN
NUM_IMAGES_PER_CLASS = 1000  # Ajusta según necesites
CLASSES = ['rojo', 'verde', 'azul']

# Crear directorios
for class_name in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

# Funciones para generar variaciones
def add_noise(image, intensity=0.1):
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy_img = np.clip(image + noise, 0, 1)
    return noisy_img

def add_blur(image, sigma=1.0):
    return gaussian_filter(image, sigma=(sigma, sigma, 0))

def adjust_brightness(image, factor):
    return np.clip(image * factor, 0, 1)

def add_gradient(image, direction='horizontal', intensity=0.5):
    h, w, _ = image.shape
    if direction == 'horizontal':
        gradient = np.tile(np.linspace(0, 1, w), (h, 1))
    else:  # vertical
        gradient = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    
    gradient = np.stack([gradient] * 3, axis=2) * intensity
    return np.clip(image + gradient, 0, 1)

def add_shapes(image, num_shapes=5):
    canvas = image.copy()
    h, w, _ = canvas.shape
    
    for _ in range(num_shapes):
        # Elegir tipo de forma (círculo, rectángulo o triángulo)
        shape_type = random.choice(['circle', 'rectangle', 'triangle'])
        
        # Color con variación de la clase pero ligeramente diferente
        base_color = canvas[h//2, w//2, :]
        color_variation = base_color * random.uniform(0.5, 1.5)
        color = np.clip(color_variation, 0, 1)
        
        # Parámetros aleatorios para la forma
        center_x = random.randint(0, w)
        center_y = random.randint(0, h)
        size = random.randint(10, 40)
        
        if shape_type == 'circle':
            cv2.circle(
                canvas, 
                (center_x, center_y), 
                size, 
                (float(color[0]), float(color[1]), float(color[2])), 
                thickness=random.choice([-1, random.randint(1, 3)])
            )
        elif shape_type == 'rectangle':
            pt1 = (max(0, center_x - size), max(0, center_y - size))
            pt2 = (min(w, center_x + size), min(h, center_y + size))
            cv2.rectangle(
                canvas, 
                pt1, 
                pt2, 
                (float(color[0]), float(color[1]), float(color[2])), 
                thickness=random.choice([-1, random.randint(1, 3)])
            )
        else:  # triangle
            pts = np.array([
                [center_x, max(0, center_y - size)],
                [max(0, center_x - size), min(h, center_y + size)],
                [min(w, center_x + size), min(h, center_y + size)]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(
                canvas, 
                [pts], 
                True, 
                (float(color[0]), float(color[1]), float(color[2])), 
                thickness=random.randint(1, 3)
            )
    
    return canvas

def add_texture(image, texture_type='noise'):
    if texture_type == 'noise':
        texture = np.random.rand(*image.shape) * 0.2
    elif texture_type == 'grid':
        h, w, _ = image.shape
        grid_size = random.randint(5, 20)
        texture = np.zeros_like(image)
        for i in range(0, h, grid_size):
            cv2.line(texture, (0, i), (w, i), (0.2, 0.2, 0.2), 1)
        for j in range(0, w, grid_size):
            cv2.line(texture, (j, 0), (j, h), (0.2, 0.2, 0.2), 1)
    elif texture_type == 'stripes':
        h, w, _ = image.shape
        texture = np.zeros_like(image)
        stripe_width = random.randint(5, 15)
        for i in range(0, h, stripe_width*2):
            cv2.rectangle(texture, (0, i), (w, i+stripe_width), (0.2, 0.2, 0.2), -1)
    
    return np.clip(image + texture, 0, 1)

def create_base_color_image(color_name, size=IMAGE_SIZE):
    img = np.zeros((size[0], size[1], 3), dtype=np.float32)
    
    if color_name == 'rojo':
        # Tonos de rojo con variaciones
        r = random.uniform(0.6, 1.0)  # Componente R alto
        g = random.uniform(0, 0.3)    # Bajo G
        b = random.uniform(0, 0.3)    # Bajo B
        img[:, :] = [r, g, b]
    
    elif color_name == 'verde':
        # Tonos de verde con variaciones
        r = random.uniform(0, 0.3)    # Bajo R
        g = random.uniform(0.6, 1.0)  # Componente G alto
        b = random.uniform(0, 0.3)    # Bajo B
        img[:, :] = [r, g, b]
    
    elif color_name == 'azul':
        # Tonos de azul con variaciones
        r = random.uniform(0, 0.3)    # Bajo R
        g = random.uniform(0, 0.3)    # Bajo G
        b = random.uniform(0.6, 1.0)  # Componente B alto
        img[:, :] = [r, g, b]
    
    return img

# Generar imágenes para cada clase
for class_name in CLASSES:
    print(f"Generando imágenes para la clase: {class_name}")
    
    for i in tqdm(range(NUM_IMAGES_PER_CLASS)):
        # Crear imagen base del color
        base_img = create_base_color_image(class_name)
        
        # Aplicar transformaciones aleatorias
        img = base_img.copy()
        
        # Seleccionar aleatoriamente transformaciones
        transformations = [
            lambda x: add_noise(x, intensity=random.uniform(0.05, 0.2)),
            lambda x: add_blur(x, sigma=random.uniform(0.5, 2.0)),
            lambda x: adjust_brightness(x, factor=random.uniform(0.7, 1.3)),
            lambda x: add_gradient(x, direction=random.choice(['horizontal', 'vertical']), 
                                 intensity=random.uniform(0.1, 0.4)),
            lambda x: add_shapes(x, num_shapes=random.randint(1, 8)),
            lambda x: add_texture(x, texture_type=random.choice(['noise', 'grid', 'stripes']))
        ]
        
        # Aplicar un número aleatorio de transformaciones
        num_transforms = random.randint(2, 5)
        for _ in range(num_transforms):
            transform = random.choice(transformations)
            img = transform(img)
        
        # Convertir a formato de imagen (0-255) y guardar
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Guardar en BGR (formato de OpenCV)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(OUTPUT_DIR, class_name, f"{class_name}_{i:04d}.jpg")
        cv2.imwrite(output_path, img_bgr)
    
    print(f"Completado: {NUM_IMAGES_PER_CLASS} imágenes generadas para {class_name}")

# Mostrar algunas imágenes de ejemplo
def show_examples():
    plt.figure(figsize=(15, 5))
    for i, class_name in enumerate(CLASSES):
        # Seleccionar 3 imágenes aleatorias de cada clase
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        image_files = os.listdir(class_dir)[:3]
        
        for j, img_file in enumerate(image_files):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3, 3, i*3 + j + 1)
            plt.imshow(img)
            plt.title(f"{class_name} - Ejemplo {j+1}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ejemplos_dataset.png"))
    plt.show()

# Mostrar estadísticas del dataset generado
def dataset_stats():
    print("\nEstadísticas del Dataset:")
    total_images = 0
    
    for class_name in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        num_images = len(os.listdir(class_dir))
        total_images += num_images
        print(f"  - Clase '{class_name}': {num_images} imágenes")
    
    print(f"Total de imágenes: {total_images}")
    print(f"Tamaño de imagen: {IMAGE_SIZE}")
    print(f"Directorio del dataset: {os.path.abspath(OUTPUT_DIR)}")

# Mostrar ejemplos y estadísticas
show_examples()
dataset_stats()

print("\n¡Dataset generado exitosamente!")
import cv2
import os
import numpy as np

# Configuración
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# Función para ordenar puntos (copiada de pruebav.py)
def order_points(points):
    """Ordena los puntos en: [sup-izq, sup-der, inf-der, inf-izq]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[1] = points[np.argmin(s)]  
    rect[3] = points[np.argmax(s)]  
    
    diff = np.diff(points, axis=1)
    rect[2] = points[np.argmin(diff)]  
    rect[0] = points[np.argmax(diff)]  
    return rect

# Añadir esta función de pruebav.py
def get_inner_contour(contour, margin_px=20):
    """Reduce el contorno hacia adentro para obtener el borde interior"""
    # Aproximar el contorno a un polígono
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Contraer el polígono
    center = np.mean(approx, axis=0)[0]
    inner_contour = []
    for point in approx:
        vector = point[0] - center
        unit_vector = vector / np.linalg.norm(vector)
        new_point = point[0] - unit_vector * margin_px  # Contraer hacia adentro
        inner_contour.append([new_point])
    
    return np.array(inner_contour, dtype=np.int32)

def apply_perspective_transform(img):
    # Preprocesamiento de la imagen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Usar equalizeHist como en pruebav.py
    equalized = cv2.equalizeHist(blurred)
    edges = cv2.Canny(equalized, 170, 180)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    if not contours:
        print("No se encontró el tablero de ajedrez en la imagen")
        return None
    
    # Obtener el contorno interior (como en pruebav.py)
    largest_contour = max(contours, key=cv2.contourArea)
    inner_contour = get_inner_contour(largest_contour, margin_px=17)
    
    # Aproximar el contorno interior a un cuadrilátero
    epsilon = 0.1 * cv2.arcLength(inner_contour, True)
    approx = cv2.approxPolyDP(inner_contour, epsilon, True)
    
    if len(approx) != 4:
        print("No se detectó un tablero de ajedrez (no es un cuadrilátero)")
        return None
    
    # Ordenar puntos y aplicar transformación de perspectiva
    points = np.array([point[0] for point in approx], dtype="float32")
    ordered_points = order_points(points)
    
    width, height = 400, 400  # Tamaño del tablero de salida
    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(ordered_points, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped

def extract_squares(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {img_path}")
        return
    
    warped = apply_perspective_transform(img)
    if warped is None:
        print(f"No se pudo procesar la imagen {img_path}")
        return
    
    # El resto del código se mantiene igual...
    
    # Extraer pieza y posición del nombre del archivo
    filename = os.path.basename(img_path)
    
    if "vacio" in filename:
        # Procesar tablero vacío
        for row in range(8):
            for col in range(8):
                square = warped[row*50:(row+1)*50, col*50:(col+1)*50]
                color = "blanco" if (row + col) % 2 == 0 else "negro"
                os.makedirs(f"{output_dir}/vacio_{color}", exist_ok=True)
                cv2.imwrite(f"{output_dir}/vacio_{color}/{row}_{col}.png", square)
    else:
        # Procesar imagen con pieza
        parts = filename.split('_')
        if len(parts) >= 3:
            piece_name = f"{parts[0]}_{parts[1]}"
            pos = parts[2]
            
            # Convertir posición a coordenadas
            col = ord(pos[0]) - ord('a')
            row = 8 - int(pos[1])
            
            # Extraer casilla con margen de seguridad
            margin = 2  # Pequeño margen para evitar bordes
            y1, y2 = max(0, row*50-margin), min(400, (row+1)*50+margin)
            x1, x2 = max(0, col*50-margin), min(400, (col+1)*50+margin)
            square = warped[y1:y2, x1:x2]
            
            os.makedirs(f"{output_dir}/{piece_name}", exist_ok=True)
            cv2.imwrite(f"{output_dir}/{piece_name}/{pos}.png", square)

# Procesar todas las fotos
for img_file in os.listdir("fotos_crudas"):
    if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        extract_squares(f"fotos_crudas/{img_file}")

print("¡Procesamiento completado!")
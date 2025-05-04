import cv2
import numpy as np
import os
from datetime import datetime

# Configuración
output_dir = "fotos_crudas"
output_dir_transformadas = "fotos_transformadas"  # Carpeta para las perspectivas corregidas
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_transformadas, exist_ok=True)
cap = cv2.VideoCapture(2)  # Ajusta el índice de tu cámara

# Variables para la transformación de perspectiva
prev_points = None
alpha = 0.2
board_size = 600  # Tamaño del tablero transformado

# Contador global para nombres únicos
photo_count = 0

print("""
¡Captura de fotos para el dataset de ajedrez!
-------------------------------------------------
Instrucciones:
1. Coloca UNA pieza en el tablero.
2. Presiona 's' para guardar la foto.
3. Ingresa el tipo de pieza y posición (ej: 'rey_blanco_a1').
4. Presiona 'q' para salir.
-------------------------------------------------
""")

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized

def get_inner_contour(contour, margin_px=20):
    """Reduce el contorno hacia adentro para obtener el borde interior"""
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    center = np.mean(approx, axis=0)[0]
    inner_contour = []
    for point in approx:
        vector = point[0] - center
        unit_vector = vector / np.linalg.norm(vector)
        new_point = point[0] - unit_vector * margin_px
        inner_contour.append([new_point])
    
    return np.array(inner_contour, dtype=np.int32)

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

def get_perspective_transform(frame):
    """Obtiene la transformación de perspectiva del tablero"""
    global prev_points
    
    processed = preprocess_image(frame)
    edges = cv2.Canny(processed, 150, 200)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    inner_contour = get_inner_contour(largest_contour, margin_px=36)
    
    epsilon = 0.1 * cv2.arcLength(inner_contour, True)
    approx = cv2.approxPolyDP(inner_contour, epsilon, True)
    
    if len(approx) != 4:
        return None
    
    points = np.array([point[0] for point in approx], dtype="float32")
    current_points = order_points(points)
    
    if prev_points is None:
        prev_points = current_points
    else:
        ordered_points = prev_points * (1 - alpha) + current_points * alpha
        prev_points = ordered_points
    
    dst = np.array([[0, 0], [board_size-1, 0], 
                   [board_size-1, board_size-1], [0, board_size-1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(prev_points, dst)
    warped = cv2.warpPerspective(frame, M, (board_size, board_size))
    
    # Mejorar contraste
    warped_yuv = cv2.cvtColor(warped, cv2.COLOR_BGR2YUV)
    warped_yuv[:,:,0] = cv2.equalizeHist(warped_yuv[:,:,0])
    warped_enhanced = cv2.cvtColor(warped_yuv, cv2.COLOR_YUV2BGR)
    
    return warped_enhanced

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo acceder a la cámara.")
        break

    # Obtener la vista transformada del tablero
    warped_board = get_perspective_transform(frame)
    
    # Mostrar ambas vistas
    cv2.imshow("Original - Presiona 's' para guardar, 'q' para salir", frame)
    if warped_board is not None:
        cv2.imshow("Tablero Transformado", warped_board)
    
    key = cv2.waitKey(1)

    if key == ord('s'):
        if warped_board is None:
            print("Error: No se detectó el tablero. Asegúrate de que sea visible.")
            continue
            
        piece_info = input("Tipo de pieza y posición (ej: 'rey_blanco_a1'): ").strip()
        
        # Generar nombre único con timestamp y contador
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{piece_info}_{timestamp}_{photo_count}"
        
        # Guardar imagen original
        cv2.imwrite(os.path.join(output_dir, filename + ".jpg"), frame)
        
        # Guardar imagen transformada
        cv2.imwrite(os.path.join(output_dir_transformadas, filename + "_transformada.jpg"), warped_board)
        
        print(f"Fotos guardadas como: {filename}.jpg y {filename}_transformada.jpg")
        photo_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n¡Listo! Se guardaron {photo_count} pares de fotos en '{output_dir}' y '{output_dir_transformadas}'.")
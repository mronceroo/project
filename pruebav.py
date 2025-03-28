import cv2
import numpy as np

cap = cv2.VideoCapture(2)  

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Gray scale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #GaussianBlur, reduce noise
    equalized = cv2.equalizeHist(blurred) #better contrast
    return equalized

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

def draw_chessboard_cells(warped_board):
    square_size = 50  # 400px / 8 casillas
    for row in range(8):
        for col in range(8):
            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            
            # Dibujar líneas verdes para separar casillas
            cv2.rectangle(warped_board, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Mostrar coordenadas de ajedrez (a1, h8, etc.)
            notation = f"{chr(97 + col)}{8 - row}"  # columnas: a-h, filas: 8-1
            cv2.putText(warped_board, notation, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    return warped_board


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = preprocess_image(frame)
    edges = cv2.Canny(processed, 170, 180)
    
    #kernel = np.ones((3, 3), np.uint8)
    #dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    #
    #closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >1000]
    
    
    output = frame.copy()
    
    if contours:
       largest_contour = max(contours, key=cv2.contourArea)
       cv2.drawContours(output, [largest_contour], -1, (0, 0, 255), 3)  
       cv2.imshow("Largest Contour", output)
       
       epsilon = 0.1 * cv2.arcLength(largest_contour, True)
       approx = cv2.approxPolyDP(largest_contour, epsilon, True) 
    
       if len(approx) == 4:
           # Ordenar los puntos para la transformación de perspectiva
           points = np.array([point[0] for point in approx], dtype="float32")
           ordered_points = order_points(points)
           
           # Dibujamos los puntos ordenados
           for i, (x, y) in enumerate(ordered_points):
               cv2.circle(output, (int(x), int(y)), 5, (0, 255, 0), -1)
               cv2.putText(output, str(i), (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
           width, height = 400, 400
           dst = np.array([[0, 0], [width-1, 0], 
                          [width-1, height-1], [0, height-1]], dtype="float32")
            
           M = cv2.getPerspectiveTransform(ordered_points, dst)
           warped = cv2.warpPerspective(frame, M, (width, height))
           
           warped_with_cells = draw_chessboard_cells(warped.copy())
           cv2.imshow("Tablero con Casillas", warped_with_cells)
           
           cv2.imshow("Tablero Rectificado", warped)

    cv2.imshow("Detección", output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
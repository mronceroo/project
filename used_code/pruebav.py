import cv2
import numpy as np

prev_points = None
alpha = 0.2

cap = cv2.VideoCapture(2)  

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Gray scale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #GaussianBlur, reduce noise
    equalized = cv2.equalizeHist(blurred) #better contrast
    return equalized

def get_inner_contour(contour, margin_px=20):#obtein inner contour, adjust value every time is run
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # contract the new poligon to for the inner contourn
    center = np.mean(approx, axis=0)[0]
    inner_contour = []
    for point in approx:
        vector = point[0] - center
        unit_vector = vector / np.linalg.norm(vector)
        new_point = point[0] - unit_vector * margin_px  # contract to the inside
        inner_contour.append([new_point])
    
    return np.array(inner_contour, dtype=np.int32)

def order_points(points):
    #order points, adjust if it is need, but normally a1->3
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[1] = points[np.argmin(s)]  
    rect[3] = points[np.argmax(s)]  
    
    diff = np.diff(points, axis=1)
    rect[2] = points[np.argmin(diff)]  
    rect[0] = points[np.argmax(diff)]  
    return rect

def draw_chessboard_cells(warped_board):
    square_size = 75  # Change to fit true size, but for reference 400/8
    for row in range(8):
        for col in range(8):
            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            
            cv2.rectangle(warped_board, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Show chess coordinates (a1, h8, etc.)
            notation = f"{chr(97 + col)}{8 - row}"  # columns: a-h, rows: 8-1
            cv2.putText(warped_board, notation, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    return warped_board


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = preprocess_image(frame)
    edges = cv2.Canny(processed, 150, 200)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >1000]
    
    
    output = frame.copy()
    
    if contours:
       largest_contour = max(contours, key=cv2.contourArea)
       inner_contour = get_inner_contour(largest_contour, margin_px=36)#Adjust ever it is run
       
    
       # Draw inner contours
       cv2.drawContours(output, [inner_contour], -1, (255, 0, 0), 2)
       
       
       epsilon = 0.1 * cv2.arcLength(inner_contour, True)
       approx = cv2.approxPolyDP(inner_contour, epsilon, True) 
       
       
    
       if len(approx) == 4:
           # Order points for perspective
           points = np.array([point[0] for point in approx], dtype="float32")
           current_points = order_points(points)
           
           # temporal filter for performance, smooth movements
           if prev_points is None:
               prev_points = current_points
           else:
               ordered_points = prev_points * (1 - alpha) + current_points * alpha
               prev_points = ordered_points
           
           # Draw order points
           for i, (x, y) in enumerate(prev_points):
               cv2.circle(output, (int(x), int(y)), 5, (0, 255, 0), -1)
               cv2.putText(output, str(i), (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
           width, height = 600, 600
           dst = np.array([[0, 0], [width-1, 0], 
                          [width-1, height-1], [0, height-1]], dtype="float32")
            
           M = cv2.getPerspectiveTransform(prev_points, dst)
           warped = cv2.warpPerspective(frame, M, (width, height))
           warped_enhanced = warped.copy()
           # Improve contrast with equalization of the histogram
           warped_yuv = cv2.cvtColor(warped_enhanced, cv2.COLOR_BGR2YUV)
           warped_yuv[:,:,0] = cv2.equalizeHist(warped_yuv[:,:,0])
           warped_enhanced = cv2.cvtColor(warped_yuv, cv2.COLOR_YUV2BGR)
           
           warped_with_cells = draw_chessboard_cells(warped_enhanced.copy())
           cv2.imshow("Tablero con Casillas", warped_with_cells)

    cv2.imshow("Detección", output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
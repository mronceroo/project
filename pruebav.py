import cv2
import numpy as np

cap = cv2.VideoCapture(1)  

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Gray scale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #GaussianBlur, reduce noise
    equalized = cv2.equalizeHist(blurred) #better contrast
    return equalized

#def detect_board(frame):
#    processed = preprocess_image(frame)
#    edges = cv2.Canny(processed, 100, 1000)#detect borders
#    print(edges)
#    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#find the contours
#    print(contours)
#    largest_contour = max(contours, key=cv2.contourArea)#search the largest so that means is the board
#
#    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
#    approx = cv2.approxPolyDP(largest_contour, epsilon, True)#Calculations over the length of the board
#
#    if len(approx) == 4:
#        points = np.array([point[0] for point in approx], dtype="float32")#For perspective
#        return points
#    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("DroidCam", frame) #original image 
    
    processed = preprocess_image(frame)
    
    
    edges = cv2.Canny(processed, 170, 180)
    cv2.imshow("Edges", edges)
    
    #kernel = np.ones((3, 3), np.uint8)
    #dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    #
    #closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >1000]
    
    
    if contours:
       largest_contour = max(contours, key=cv2.contourArea)
       cv2.drawContours(output, [largest_contour], -1, (0, 0, 255), 3)  
       cv2.imshow("Largest Contour", output)
        
    output = frame.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", output)
    
    #largest_contour = max(contours, key=cv2.contourArea)
    #cv2.drawContours(output, [largest_contour], -1, (0, 0, 255), 3)  
    #cv2.imshow("Largest Contour", output)
    
    #board_c= detect_board(frame)
    #if board_c is not None:
    # print(board_c)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r"C:\Users\manue\OneDrive\Documentos\Universidad\Proyecto\código\yolotry3.pt")

class_names = ['b-bishop', 'b-king', 'b-knight', 'b-pawn', 'b-queen', 'b-rook', 'w-bishop', 'w-king', 'w-knight', 'w-pawn', 'w-queen', 'w-rook']

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)  # stream=True for more efficient processing
    
    for result in results:
        if result.masks is not None:
            masks = result.masks.data
            masks = masks.cpu().numpy()
            
            # boxes and clases
            boxes = result.boxes.data.cpu().numpy()
            
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                x1, y1, x2, y2, conf, class_id = box
                class_id = int(class_id)
                
                if conf > 0.5:  
                    # bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # label
                    label = f"{class_names[class_id]} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # segmentation mask
                    mask = mask.squeeze()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
                    mask = mask.astype(np.uint8)
                    
                    color_mask = np.zeros_like(frame)
                    color_mask[mask == 1] = [0, 255, 0]
                    
                    frame = cv2.addWeighted(frame, 1, color_mask, 0.3, 0)

    cv2.imshow('Chess Piece Detection - YOLOv8 Segmentation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
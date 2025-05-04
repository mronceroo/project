import cv2
import numpy as np
import tensorflow as tf

# Load model and prepare Class names
model = tf.keras.models.load_model("Manuel_Roncero_CNN.h5")

class_names = ['triangle', 'star', 'circle']

# Camera settings
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to model's expected input size (50x50) and keep color (3 channels)
    resized = cv2.resize(frame, (50, 50))
    
    # Normalize image (convert to float32 and scale to [-1, 1])
    img = resized.astype('float32') / 127.5 - 1.0
    img = img.reshape(1, 50, 50, 3)  # Add batch dimension
    
    # Apply model to do predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    class_label = class_names[predicted_class]
    
    # Show prediction and image
    cv2.putText(frame, f"Prediction: {class_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Manuel_Roncero_CNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
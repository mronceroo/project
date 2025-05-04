import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Generación de datos sintéticos
# ----------------------------
def generate_chess_dataset(num_samples=1000, board_size=64, square_size=8):
    """Genera imágenes de tableros de ajedrez con objetos pequeños en casillas."""
    classes = ['red', 'green', 'blue']
    X = []
    y = []
    
    for _ in range(num_samples):
        # Crear tablero de ajedrez (blanco y negro)
        img = np.zeros((board_size, board_size, 3), dtype=np.float32)
        for i in range(0, board_size, square_size):
            for j in range(0, board_size, square_size):
                if (i//square_size + j//square_size) % 2 == 0:
                    img[i:i+square_size, j:j+square_size] = [1.0, 1.0, 1.0]  # Blanco
        
        # Añadir objeto pequeño (70% del tamaño de la casilla)
        class_idx = np.random.randint(0, 3)
        color = classes[class_idx]
        row = np.random.randint(0, board_size//square_size) * square_size
        col = np.random.randint(0, board_size//square_size) * square_size
        obj_size = int(square_size * 0.7)
        offset = (square_size - obj_size) // 2
        
        if color == 'red':
            img[row+offset:row+offset+obj_size, col+offset:col+offset+obj_size] = [1.0, 0.0, 0.0]
        elif color == 'green':
            img[row+offset:row+offset+obj_size, col+offset:col+offset+obj_size] = [0.0, 1.0, 0.0]
        else:  # blue
            img[row+offset:row+offset+obj_size, col+offset:col+offset+obj_size] = [0.0, 0.0, 1.0]
        
        X.append(img)
        y.append(class_idx)
    
    return np.array(X), utils.to_categorical(np.array(y), num_classes=3)

# Generar datasets
X_train, y_train = generate_chess_dataset(5000)  # 5000 ejemplos de entrenamiento
X_test, y_test = generate_chess_dataset(1000)    # 1000 ejemplos de test

# ----------------------------
# 2. Modelo CNN especializado
# ----------------------------
def create_model(input_shape=(64, 64, 3)):
    model = models.Sequential([
        # Bloque 1 - Detección de bordes y patrones locales
        layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Bloque 2 - Captura características del objeto pequeño
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Bloque 3 - Contexto global
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        
        # Clasificación
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model()
model.summary()

# ----------------------------
# 3. Entrenamiento con early stopping
# ----------------------------
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping]
)

# ----------------------------
# 4. Evaluación
# ----------------------------
# Gráfica de precisión
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Evolución de la precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Matriz de confusión
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rojo', 'Verde', 'Azul'],
            yticklabels=['Rojo', 'Verde', 'Azul'])
plt.title('Matriz de Confusión')
plt.show()

# ----------------------------
# 5. Guardar modelo y funciones de inferencia
# ----------------------------
model.save('chess_object_classifier.h5')

def predict_chess_object(img_path, model_path='chess_object_classifier.h5'):
    """Predice el color de un objeto en una imagen de tablero."""
    # Cargar modelo
    model = tf.keras.models.load_model(model_path)
    
    # Preprocesar imagen
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predecir
    pred = model.predict(img)
    return ['red', 'green', 'blue'][np.argmax(pred)]

# Ejemplo de uso:
print(predict_chess_object('azul.jpg'))
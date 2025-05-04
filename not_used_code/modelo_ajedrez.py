import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Configuración
INPUT_SHAPE = (50, 50, 3)
NUM_CLASSES = 3  # rey_negro, torre_negra, rey_blanco, etc.
BATCH_SIZE = 32
EPOCHS = 3

# Generadores de datos con aumento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

# Usar el mismo directorio para entrenamiento y validación
data_dir = "dataset/aumentado2"

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# Para validación, solo rescale
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Imprimir información de las clases
print("Clases encontradas:", train_generator.class_indices)
print(f"Total de muestras de entrenamiento: {train_generator.samples}")
print(f"Total de muestras de validación: {val_generator.samples}")

# Crear modelo con Transfer Learning
# Cargar MobileNetV2 pre-entrenado sin las capas superiores
base_model = MobileNetV2(
    input_shape=INPUT_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Congelar el modelo base
base_model.trainable = False

# Crear nuevo modelo con el modelo base
model = models.Sequential([
    # Modelo base
    base_model,
    
    # Capas nuevas para nuestra tarea específica
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    
    # Primera capa densa
    layers.Dense(128, activation=None),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    # Capa de salida
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compilar modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Mostrar resumen del modelo
model.summary()

# Callbacks para mejorar el entrenamiento
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# Entrenar el modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Guardar modelo
model.save("modelo_ajedrez_transfer.h5")

# Evaluación: precisión durante entrenamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Fine-tuning: descongelar algunas capas del modelo base y entrenar con tasa de aprendizaje baja
print("Comenzando fine-tuning...")

# Descongelar las últimas capas del modelo base
base_model.trainable = True
fine_tune_at = 100  # Número aproximado de capas a descongelar
for layer in base_model.layers[:-fine_tune_at]:
    layer.trainable = False

# Recompilar el modelo con una tasa de aprendizaje más baja
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Tasa más baja
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tuning (entrenamiento adicional)
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # Menos épocas para fine-tuning
    callbacks=callbacks,
    initial_epoch=len(history.history['loss'])  # Empezar desde la última época
)

# Guardar modelo con fine-tuning
model.save("modelo_ajedrez_transfer_finetuned.h5")

# Evaluación detallada
print("Evaluando en validación...")
val_generator.reset()
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
class_indices = {v: k for k, v in val_generator.class_indices.items()}

print("\nMatriz de confusión:")
cm = confusion_matrix(val_generator.classes, y_pred_classes)
print(cm)

print("\nInforme de clasificación:")
print(classification_report(
    val_generator.classes, 
    y_pred_classes, 
    target_names=list(class_indices.values())
))

# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(len(class_indices))
plt.xticks(tick_marks, class_indices.values(), rotation=45)
plt.yticks(tick_marks, class_indices.values())
plt.tight_layout()
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.savefig('confusion_matrix.png')
plt.show()
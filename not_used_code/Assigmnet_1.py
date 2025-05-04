import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, optimizers, losses
import matplotlib.pyplot as plt
import numpy as np
import tf2onnx
import onnx

#Preprocessing

(train_images, train_labels), (test_images, test_labels) ="dataset/piezas_ajedrez" #tf.keras.datasets.fashion_mnist.load_data()


#Normailze test and train images. Convert pixel images values to float32. /255 to change to range 0,1. Scale pixels values *2 - 1 to change range to -1,1. This change help NeuralNets to converge during training
train_images = (train_images.astype('float32') / 255) * 2 - 1
test_images = (test_images.astype('float32') / 255) * 2 - 1

class_names = ["azul", "rojo", "verde"]#["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               #"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Show some training images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = (train_images[i] + 1) / 2  # Scale back to [0,1] for display
    plt.imshow(img)
    plt.title(class_names[train_labels[i]])
    plt.axis('off')
plt.show()

#CNN, training and evaluate

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),#Conv Layer
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),#Conv Layer Number filter
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),#Conv Layer
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),#Prevent overfitting
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10)
])

optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.9)
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    epochs=64,
                    batch_size=32,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc*100:.2f}%')

#Save model with HDF5

model.save("Manuel_Roncero_CNN.h5")
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Check for GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Step 1: Image Preprocessing using OpenCV
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (256, 256))           # Resize to 256x256
    img = img / 255.0                           # Normalize pixel values
    return img

# Load dataset paths
train_data_dir = 'C:/Users/abhip/Desktop/project/train'
test_data_dir = 'C:/Users/abhip/Desktop/project/test' 

# Step 2: Data Augmentation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    brightness_range=[0.8, 1.2]  
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Determine number of classes dynamically
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'  # Default to categorical
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)  # Get number of classes
print(f"Detected {num_classes} classes")

# Set correct class_mode
class_mode = 'binary' if num_classes == 2 else 'categorical'
train_generator.class_mode = class_mode
test_generator.class_mode = class_mode

# Step 3: Class Weights for Imbalanced Data
labels = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class Weights: {class_weights_dict}")

# Step 4: Model Definition with CNN and Batch Normalization
model = models.Sequential()

# Convolutional Blocks
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# Fully Connected Layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))  

# Adjust final layer based on number of classes
if num_classes == 2:
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    loss_function = 'binary_crossentropy'
else:
    model.add(layers.Dense(num_classes, activation='softmax'))  # Multi-class classification
    loss_function = 'categorical_crossentropy'

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss=loss_function,
              metrics=['accuracy'])

# Step 6: Callbacks for Learning Rate Scheduler and Early Stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 7: Train the Model with Class Weights
history = model.fit(
    train_generator,
    epochs=50,  # Increased epochs for better learning
    validation_data=test_generator,
    class_weight=class_weights_dict,  # Handles class imbalance
    callbacks=[reduce_lr, early_stopping]
)

# Step 8: Test the Model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.4f}')

# Step 9: Save Model
model.save('leaf_disease_model.h5')

# Step 10: Plot Training and Validation Accuracy/Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Step 11: Confusion Matrix and Classification Report
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1) if num_classes > 2 else (Y_pred > 0.5).astype(int).flatten()

print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print('Classification Report')
class_labels = list(test_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_pred, target_names=class_labels))

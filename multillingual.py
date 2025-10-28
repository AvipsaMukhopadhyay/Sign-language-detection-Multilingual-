"""
Indian Sign Language (ISL) Static Image Detection
-------------------------------------------------
This script trains a CNN to classify ISL alphabet / number signs
from still images stored in folder structure:

ISL/
 └── Indian/
      ├── A/
      │   ├── img1.jpg
      │   ├── img2.jpg
      │   └── ...
      ├── B/
      │   ├── img1.jpg
      │   └── ...
      └── 1/
          ├── img1.jpg
          └── ...

Each subfolder name = one gesture label.
"""

# --------------------------
# IMPORTS
# --------------------------
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.model_selection import train_test_split

# --------------------------
# CONFIGURATION
# --------------------------
DATASET_DIR = DATASET_DIR = r"C:\Users\Avipsa\OneDrive\Desktop\Multilingual DL\ISL\Indian"
      # Path to your dataset root
IMAGE_SIZE = (64, 64)           # Resize all images to 64x64
BATCH_SIZE = 32
EPOCHS = 20                     # Increase if you have more data
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "saved_model/isl_cnn_model.h5"

os.makedirs("saved_model", exist_ok=True)

# --------------------------
# 1️⃣ LOAD AND PREPROCESS IMAGES
# --------------------------
def load_image(img_path, size=IMAGE_SIZE):
    """Reads and preprocesses one image file."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img

def build_samples_labels(dataset_dir):
    """Walk through folders and collect image paths + numeric labels."""
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    samples, labels = [], []
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for c in classes:
        cdir = os.path.join(dataset_dir, c)
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append(os.path.join(cdir, fname))
                labels.append(class_to_idx[c])
    return samples, labels, classes

print("Loading dataset...")
samples, labels, classes = build_samples_labels(DATASET_DIR)
num_classes = len(classes)
print(f"Detected {num_classes} classes:", classes)
print(f"Total images: {len(samples)}")

# --------------------------
# 2️⃣ SPLIT DATA
# --------------------------
train_samples, val_samples, train_labels, val_labels = train_test_split(
    samples, labels, test_size=0.2, stratify=labels, random_state=42
)

# --------------------------
# 3️⃣ DATA GENERATOR
# --------------------------
class ImageDatasetGenerator(tf.keras.utils.Sequence):
    """Loads images batch by batch for training."""
    def __init__(self, samples, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = [], []
        for i in batch_idx:
            img = load_image(self.samples[i])
            X.append(img)
            y.append(self.labels[i])
        X = np.array(X, dtype=np.float32)
        y = tf.keras.utils.to_categorical(np.array(y), num_classes=num_classes)
        return X, y

# Create generators
# train_gen = ImageDatasetGenerator(train_samples, train_labels, batch_size=BATCH_SIZE)
# val_gen   = ImageDatasetGenerator(val_samples, val_labels, batch_size=BATCH_SIZE, shuffle=False)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Only rescaling for validation (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)



# --------------------------
# 4️⃣ MODEL DEFINITION (CNN)
# --------------------------
def build_cnn(input_shape=(64, 64, 3), n_classes=36):
    """Simple CNN classifier for static sign images."""
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Build the model
model = build_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), n_classes=num_classes)
model.summary()

# --------------------------
# 5️⃣ TRAINING
# --------------------------
lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
ckpt_cb = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')

print("\nStarting training...\n")
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS,
                    callbacks=[lr_cb, ckpt_cb])

print("\nTraining complete. Best model saved to:", MODEL_SAVE_PATH)

# --------------------------
# 6️⃣ TEST / INFERENCE
# --------------------------
def predict_from_image(model, img_path, class_names):
    img = load_image(img_path)
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    idx = np.argmax(pred)
    label = class_names[idx]
    conf = float(np.max(pred))
    return label, conf

# Example test
test_image = val_samples[0]  # just one validation image
true_label = classes[val_labels[0]]
pred_label, conf = predict_from_image(model, test_image, classes)

print("\nTest image:", test_image)
print(f"True label: {true_label}")
print(f"Predicted:  {pred_label} (confidence {conf:.2f})")

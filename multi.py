
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import json
import random

DATASET_DIR_ISL = r"C:\Users\Avipsa\OneDrive\Desktop\Multilingual DL\ISL\Indian"
DATASET_DIR_WLASL = r"C:\Users\Avipsa\Downloads\archive\dataset\SL"

IMG_SIZE = 64
MAX_FRAMES = 8
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WLASL_SAMPLES = 100 

os.makedirs("saved_model", exist_ok=True)

def load_image(path, size=(IMG_SIZE, IMG_SIZE)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img


def extract_frames(video_path, max_frames=MAX_FRAMES, size=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            frames.append(frame.astype(np.float32) / 255.0)
        count += 1
    cap.release()
    while len(frames) < max_frames:
        frames.append(np.zeros((size[0], size[1], 3), dtype=np.float32))
    return np.array(frames[:max_frames])

def build_isl_samples(dataset_dir):
    classes = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])
    samples, labels = [], []
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for c in classes:
        cdir = os.path.join(dataset_dir, c)
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append(os.path.join(cdir, fname))
                labels.append(class_to_idx[c])
    return samples, labels, classes

print(" Loading ISL dataset...")
image_samples, image_labels, isl_classes = build_isl_samples(DATASET_DIR_ISL)
print(f"ISL: {len(image_samples)} images | {len(isl_classes)} classes")

def build_wlasl_samples(dataset_dir, max_samples=NUM_WLASL_SAMPLES):
    folders = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])
    samples, labels = [], []
    class_to_idx = {c: i + len(isl_classes) for i, c in enumerate(folders)}

    for c in folders:
        cdir = os.path.join(dataset_dir, c)
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".mp4", ".avi", ".mov")):
                samples.append(os.path.join(cdir, fname))
                labels.append(class_to_idx[c])
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break
    return samples, labels, folders

print("📂 Loading WLASL dataset...")
video_samples, video_labels, wlasl_classes = build_wlasl_samples(DATASET_DIR_WLASL)
print(f"✅ WLASL: {len(video_samples)} videos | {len(wlasl_classes)} classes")
all_classes = isl_classes + wlasl_classes
num_classes = len(all_classes)
print(f"🌍 Total combined classes: {num_classes}")

with open("saved_model/label_mapping.json", "w") as f:
    json.dump(all_classes, f)

def image_generator():
    for path, label in zip(image_samples, image_labels):
        try:
            img = load_image(path)
            yield (img, label)
        except Exception as e:
            print("⚠️ Error loading image:", path, e)

def video_generator():
    for path, label in zip(video_samples, video_labels):
        try:
            frames = extract_frames(path)
            yield (frames, label)
        except Exception as e:
            print("⚠️ Error loading video:", path, e)

image_ds = tf.data.Dataset.from_generator(
    image_generator,
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

video_ds = tf.data.Dataset.from_generator(
    video_generator,
    output_signature=(
        tf.TensorSpec(shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

def build_cnn_lstm(input_shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3), n_classes=num_classes):
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu'), input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D(2,2)),
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D(2,2)),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(128),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model
image_model = build_cnn()
video_model = build_cnn_lstm()

image_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])
video_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

print("\n🚀 Training ISL image model...")
image_model.fit(image_ds, epochs=EPOCHS)

print("\n🎥 Training WLASL video model...")
video_model.fit(video_ds, epochs=EPOCHS)

# Save models
image_model.save("saved_model/isl_image_model.h5")
video_model.save("saved_model/wlasl_video_model.h5")
print("\n Training complete. Models saved!")

def predict_image(model, img_path):
    img = load_image(img_path)
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    idx = np.argmax(pred)
    return all_classes[idx], float(np.max(pred))

def predict_video(model, video_path):
    frames = extract_frames(video_path)
    pred = model.predict(np.expand_dims(frames, axis=0), verbose=0)[0]
    idx = np.argmax(pred)
    return all_classes[idx], float(np.max(pred))

print("\n🔍 Testing predictions...\n")
test_img_path = random.choice(image_samples)
true_img_label = isl_classes[image_labels[image_samples.index(test_img_path)]]
pred_img_label, conf_img = predict_image(image_model, test_img_path)
print(f"🖼 ISL → True: {true_img_label} | Predicted: {pred_img_label} | Confidence: {conf_img:.2f}")

test_vid_path = random.choice(video_samples)
true_vid_label = wlasl_classes[video_labels[video_samples.index(test_vid_path)] - len(isl_classes)]
pred_vid_label, conf_vid = predict_video(video_model, test_vid_path)
print(f"🎬 WLASL → True: {true_vid_label} | Predicted: {pred_vid_label} | Confidence: {conf_vid:.2f}")


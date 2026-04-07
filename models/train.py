import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
LayerNormalization = tf.keras.layers.LayerNormalization
Xception = tf.keras.applications.Xception
Adam = tf.keras.optimizers.Adam
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
l2 = tf.keras.regularizers.l2

# ---------------------------
# 📁 PATHS (LOCAL)
# ---------------------------
base_path = r"dataset/Pediatric Chest X-ray Pneumonia/train"
base_path_test = r"dataset/Pediatric Chest X-ray Pneumonia/test"

# ---------------------------
# 📊 LOAD DATA
# ---------------------------
input_path = []
label = []

for class_name in os.listdir(base_path):
    for file in os.listdir(os.path.join(base_path, class_name)):
        input_path.append(os.path.join(base_path, class_name, file))
        label.append(0 if class_name == 'NORMAL' else 1)

df = pd.DataFrame({"image": input_path, "label": label})
df = df.sample(frac=1).reset_index(drop=True)

# ---------------------------
# 📊 TEST DATA
# ---------------------------
input_path_test = []
label_test = []

for class_name in os.listdir(base_path_test):
    for file in os.listdir(os.path.join(base_path_test, class_name)):
        input_path_test.append(os.path.join(base_path_test, class_name, file))
        label_test.append(0 if class_name == 'NORMAL' else 1)

df_test = pd.DataFrame({"image": input_path_test, "label": label_test})
df_test = df_test.sample(frac=1).reset_index(drop=True)

# ---------------------------
# 🔀 SPLIT DATA
# ---------------------------
df['label'] = df['label'].astype(str)

train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.4, random_state=42)

# ---------------------------
# 🖼️ IMAGE GENERATORS
# ---------------------------
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2)
val_gen = ImageDataGenerator(rescale=1./255)

train_iterator = train_gen.flow_from_dataframe(
    train, x_col='image', y_col='label',
    target_size=(224, 224), batch_size=8,
    class_mode='binary'
)

val_iterator = val_gen.flow_from_dataframe(
    val, x_col='image', y_col='label',
    target_size=(224, 224), batch_size=8,
    class_mode='binary'
)

# ---------------------------
# ⚖️ CLASS WEIGHTS
# ---------------------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train['label']),
    y=train['label']
)
class_weight_dict = dict(enumerate(class_weights))

# ---------------------------
# 🧠 MODEL
# ---------------------------
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= len(base_model.layers) - 25)

model = tf.keras.models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    LayerNormalization(),
    Dropout(0.25),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    LayerNormalization(),
    Dropout(0.25),

    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    LayerNormalization(),
    Dropout(0.25),

    Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
    LayerNormalization(),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# 🚀 TRAIN (SHORT FOR NOW)
# ---------------------------
history = model.fit(
    train_iterator,
    epochs=3,   # 🔥 keep small for now
    validation_data=val_iterator,
    class_weight=class_weight_dict
)

# ---------------------------
# 💾 SAVE MODEL (IMPORTANT)
# ---------------------------
model.save("models/pneumonia_model.keras")

print("✅ MODEL SAVED SUCCESSFULLY")
import os
import random
import shutil

# Configura tus rutas aquí
images_dir = '/home/marco/Escritorio/Yolo/imgs/train/trainAll'   # Pasta con TODAS las imágenes
labels_dir = '/home/marco/Escritorio/Yolo/labels/train/trainall'   # Pasta con TODAS las etiquetas

train_images_dir = '/home/marco/Escritorio/Yolo/yoloTrain/img'
val_images_dir   = '/home/marco/Escritorio/Yolo/YoloVal/img'
train_labels_dir = '/home/marco/Escritorio/Yolo/yoloTrain/label'
val_labels_dir   = '/home/marco/Escritorio/Yolo/YoloVal/label'

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

val_ratio = 0.12  # 20% validación

# Lista las imágenes (ajusta la extensión si fuera .jpeg)
images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(images)

val_count = int(len(images) * val_ratio)
val_images = set(images[:val_count])

for img_file in images:
    label_file = os.path.splitext(img_file)[0] + '.txt'
    src_img = os.path.join(images_dir, img_file)
    src_lbl = os.path.join(labels_dir, label_file)

    if img_file in val_images:
        dst_img = os.path.join(val_images_dir, img_file)
        dst_lbl = os.path.join(val_labels_dir, label_file)
    else:
        dst_img = os.path.join(train_images_dir, img_file)
        dst_lbl = os.path.join(train_labels_dir, label_file)

    shutil.copy2(src_img, dst_img)
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)

print("Separación terminada: train y val.")
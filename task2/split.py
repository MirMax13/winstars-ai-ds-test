import os
import shutil
import random
from translate import translate

RAW_DIR = "./task2/raw-img"
OUTPUT_DIR = "./task2/dataset"
TRAIN_RATIO = 0.8  # 80% train, 20% test

for split in ["train", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    english_name = translate.get(class_name, class_name)

    for split in ["train", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, english_name), exist_ok=True)

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for img_list, split in [(train_imgs, "train"), (test_imgs, "test")]:
        for img_name in img_list:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(OUTPUT_DIR, split, english_name, img_name)
            shutil.copy(src, dst)

print("✅ Dataset successfully split into 'train' and 'test' folders.")
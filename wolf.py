import cv2
import numpy as np
import pandas as pd
import os
from glob import glob
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ===== НАЛАШТУВАННЯ =====
folder = r"C:\Users\kater\Desktop\лаби\solar_data"

k = 1.0
min_spot_area = 3
group_distance = 45


# ===== ФУНКЦІЯ ОБРОБКИ ОДНОГО ЗНІМКА =====
from PIL import Image

def calculate_wolf_number(image_path):

    try:
        pil_img = Image.open(image_path).convert("RGB")
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Помилка читання:", image_path)
        print(e)
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ===== МАСКА ДИСКА СОНЦЯ =====
    # Відкидаємо чорний фон навколо Сонця
    sun_mask = gray > 40

    # Трохи чистимо маску
    sun_mask = sun_mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    sun_mask = cv2.morphologyEx(sun_mask, cv2.MORPH_CLOSE, kernel)

    # ===== ПОШУК ПЛЯМ ВСЕРЕДИНІ ДИСКА =====
    sun_pixels = gray[sun_mask > 0]

    if len(sun_pixels) == 0:
        return None

    # Поріг: плями темніші за середню яскравість Сонця
    threshold_value = np.mean(sun_pixels) - 35

    spots_mask = np.zeros_like(gray)
    spots_mask[(gray < threshold_value) & (sun_mask > 0)] = 255

    # Прибираємо шум
    spots_mask = spots_mask.astype(np.uint8)
    spots_mask = cv2.morphologyEx(spots_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(
        spots_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    spots = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Відкидаємо занадто дрібний шум і занадто великі об'єкти
        if 2 <= area <= 500:
            M = cv2.moments(cnt)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                spots.append([cx, cy])

    s = len(spots)

    if s == 0:
        g = 0
        W = 0
    else:
        spots_array = np.array(spots)

        clustering = DBSCAN(
            eps=30,
            min_samples=1
        ).fit(spots_array)

        g = len(set(clustering.labels_))
        W = k * (10 * g + s)

    return {
        "file": os.path.basename(image_path),
        "groups_g": g,
        "spots_s": s,
        "wolf_number_W": W
    }


# ===== ПОШУК УСІХ ЗНІМКІВ =====
image_files = sorted(
    glob(os.path.join(folder, "*.jpg")) +
    glob(os.path.join(folder, "*.jpeg")) +
    glob(os.path.join(folder, "*.png")) +
    glob(os.path.join(folder, "*.tif")) +
    glob(os.path.join(folder, "*.tiff")) +
    glob(os.path.join(folder, "*.jp2"))
)

print("Знайдено файлів:", len(image_files))

if len(image_files) == 0:
    print("Файли не знайдено. Перевір шлях до папки.")
    exit()


# ===== ОБРОБКА ВСІХ ЗНІМКІВ =====
results = []

for image_path in image_files:
    result = calculate_wolf_number(image_path)

    if result is not None:
        results.append(result)

df = pd.DataFrame(results)

print("Перші рядки таблиці:")
print(df.head())


# ===== ДОДАЄМО ДАТУ З НАЗВИ ФАЙЛУ =====
df["date"] = df["file"].str.extract(r"(\d{8})")
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

df = df.sort_values("date")


# ===== ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ =====
df.to_csv("wolf_numbers.csv", index=False, encoding="utf-8-sig")

print("Результати збережено у файл wolf_numbers.csv")


# ===== ПОБУДОВА ГРАФІКА =====
plt.figure(figsize=(12, 5))

plt.plot(
    df["date"],
    df["wolf_number_W"],
    marker="o",
    linewidth=1
)

plt.xlabel("Дата")
plt.ylabel("Число Вольфа W")
plt.title("Зміна числа Вольфа за знімками Сонця")
plt.grid(True)

plt.show()
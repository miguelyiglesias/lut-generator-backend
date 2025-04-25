import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil

app = FastAPI()

# Crear carpeta local segura
LOCAL_DATA_DIR = "./generated_files"
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(LOCAL_DATA_DIR, "uploads"), exist_ok=True)

# Montar carpeta pública para servir archivos
app.mount("/static", StaticFiles(directory=LOCAL_DATA_DIR), name="static")

# Funciones auxiliares
def extract_color_profile(image_path: str, num_clusters: int = 5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pixels)
    return kmeans.cluster_centers_

def average_color_profiles(profiles):
    all_colors = np.vstack(profiles)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(all_colors)
    return kmeans.cluster_centers_

def match_color_to_palette(color, palette):
    distances = np.linalg.norm(palette - color, axis=1)
    return palette[np.argmin(distances)].astype(np.uint8)

def generate_lut(reference_colors, lut_size=33):
    lut = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.uint8)
    for r in range(lut_size):
        for g in range(lut_size):
            for b in range(lut_size):
                color = np.array([r, g, b]) * 255 / (lut_size - 1)
                lut[r, g, b] = match_color_to_palette(color, reference_colors)
    return lut

def save_lut_as_cube(lut, filename="generated_lut.cube"):
    path = os.path.join(LOCAL_DATA_DIR, filename)
    with open(path, "w") as f:
        f.write("TITLE \"Generated LUT\"\n")
        f.write(f"LUT_3D_SIZE {lut.shape[0]}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for b in range(lut.shape[0]):
            for g in range(lut.shape[1]):
                for r in range(lut.shape[2]):
                    rgb = lut[r, g, b] / 255.0
                    f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
    return path

@app.post("/generate-lut/")
async def generate_lut_from_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...)
):
    upload_dir = os.path.join(LOCAL_DATA_DIR, "uploads")
    paths = []
    for idx, image in enumerate([image1, image2, image3]):
        path = os.path.join(upload_dir, f"image_{idx}.jpg")
        with open(path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        paths.append(path)

    color_profiles = [extract_color_profile(p) for p in paths]
    average_palette = average_color_profiles(color_profiles)
    lut = generate_lut(average_palette)
    lut_path = save_lut_as_cube(lut)

    download_url = f"https://lut-generator-backend.onrender.com/static/{os.path.basename(lut_path)}"
    return JSONResponse(content={"download_url": download_url})

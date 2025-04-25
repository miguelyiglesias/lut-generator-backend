import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil

# Inicializar FastAPI
app = FastAPI()

# Crear carpeta si no existe (para evitar errores)
os.makedirs("/mnt/data", exist_ok=True)

# Montar carpeta pública para servir archivos .cube
app.mount("/static", StaticFiles(directory="/mnt/data"), name="static")

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
        for g in range(lut_size_

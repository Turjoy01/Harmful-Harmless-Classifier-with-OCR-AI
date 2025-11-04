# main.py
import os
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import tensorflow as tf
from google.cloud import vision

# ==================== CONFIG ====================
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-key.json"

# ==================== LOAD .pkl ====================
with open("best_harmful_classifier.pkl", "rb") as f:
    harmful_classifier = pickle.load(f)

tokenizer = harmful_classifier['tokenizer']
max_len = harmful_classifier['max_len']
model_architecture = harmful_classifier['model_architecture']
model_weights = harmful_classifier['model_weights']
label_mapping = harmful_classifier['label_mapping']

model = tf.keras.models.model_from_json(model_architecture)
model.set_weights(model_weights)

# ==================== OCR ====================
def ocr_google(image_bytes: bytes) -> str:
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    resp = client.text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.text_annotations[0].description if resp.text_annotations else ""

# ==================== EXTRACT TEXT (ROBUST) ====================
def extract_text(full_ocr: str):
    lines = [ln.strip() for ln in full_ocr.splitlines() if ln.strip()]
    if not lines:
        return "UNKNOWN", ""

    product_name = lines[0]

    # --- CASE 1: "INGREDIENTS" exists ---
    ingredients_lines = []
    capture = False
    for i, line in enumerate(lines):
        if "INGREDIENTS" in line.upper():
            capture = True
            remainder = line.split("INGREDIENTS:", 1)[-1].strip()
            if remainder:
                ingredients_lines.append(remainder)
            start_idx = i + 1
            break
    else:
        start_idx = 1  # fallback: start after product name

    # --- CASE 2: Capture next 3–5 lines (fallback) ---
    if not capture:
        ingredients_lines = lines[start_idx:start_idx + 5]
    else:
        # capture until breaker or max 5 lines
        for line in lines[start_idx:start_idx + 5]:
            up = line.upper()
            if any(stop in up for stop in ["NUTRITION", "ALLERGEN", "DIRECTIONS", "NET WT"]):
                break
            ingredients_lines.append(line)

    # --- Clean & join ---
    raw_text = " ".join(ingredients_lines)
    cleaned = " ".join(raw_text.lower().split())
    return product_name, cleaned

# ==================== CLASSIFY ====================
def classify_text(text: str):
    if not text.strip():
        return "unknown", 0.0

    tokenized = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=max_len)

    probs = model.predict(padded, verbose=0)
    idx = tf.argmax(probs, axis=1).numpy()[0]
    pred = label_mapping[idx]
    conf = float(probs[0][idx])
    return pred, round(conf, 4)

# ==================== FASTAPI ====================
app = FastAPI()

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        Image.open(BytesIO(contents)).convert("RGB")
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    full_ocr = ocr_google(contents)
    if not full_ocr.strip():
        return JSONResponse(status_code=400, content={"error": "No text"})

    product_name, extracted_text = extract_text(full_ocr)
    prediction, confidence = classify_text(extracted_text)

    return {
        "product_name": product_name,
        "extracted_text": extracted_text,
        "classification": {
            "prediction": prediction,
            "confidence": confidence
        }
    }

@app.get("/")
async def root():
    return {"message": "ROBUST OCR + Classification API"}
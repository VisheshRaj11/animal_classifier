from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from model_loader import get_model
from utils import read_image
import tensorflow as tf
import numpy as np
import os
import uvicorn
from med import get_medicine_suggestion  # 🔗 Link to Gemini

app = FastAPI()

CLASS_NAMES = {
    "dog": ["Demodicosis", "Dermatitits", "Fungal_Infection", "Healthy", "Hypersenstivity", "Ringworm"],
    "cat": ["Flea_Allergy", "Health", "Ringworm", "Scabies"],
    "cow": ["Lumpy Skin", "Normal Skin"]
}

@app.post("/predict/")
async def predict(
    animal_type: str = Form(...),
    file: UploadFile = File(...)
):
    if animal_type not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid animal_type. Choose from dog, cat, cow.")

    image_bytes = await file.read()
    img_array = read_image(image_bytes)

    # Load the appropriate model based on the animal type
    # if animal_type == "dog":
    #     model = tf.keras.models.load_model("Dog.keras")
    # elif animal_type == "cat":
    #     model = tf.keras.models.load_model("cat.keras")
    # else:
    #     model = tf.keras.models.load_model("Lumpy.keras")
    model = get_model(animal_type)  # Lazy load model
    # Predict the label
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    label_index = np.argmax(prediction)
    label = CLASS_NAMES[animal_type][label_index]

    # Normalize label for healthy cases
    if label in ["Health", "Lumpy Skin"]:
        label = "Healthy"
    if label in ["Flea_Allergy"]:
        label = "Flea Allergy"
    if label in ["Fungal_infections"]:
        label = "Fungal infections"
    # 💊 Get medicine suggestion from Gemini 2.5 Flash
    try:
        medicine_suggestion = get_medicine_suggestion(label, animal_type)
    except Exception as e:
        medicine_suggestion = f"Failed to fetch suggestion: {str(e)}"

    return JSONResponse(content={
        "prediction": label,
        "medicine_suggestion": medicine_suggestion
    })

from fastapi import FastAPI, File, UploadFile
from inference import predict_image
import shutil

# Initialize FastAPI app
app = FastAPI()

# Define the prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file locally
    file_location = f"./uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get the prediction from the model
    prediction = predict_image(file_location)

    # Return prediction as JSON response
    return {"filename": file.filename, "prediction": int(prediction[0])}

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_image
import os
import shutil
import time

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for storing uploaded files
UPLOAD_DIR = "./uploads/"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Define the prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file locally with a unique name
        timestamp = int(time.time())  # Use current timestamp for unique filename
        file_extension = os.path.splitext(file.filename)[1]  # Get file extension
        unique_filename = f"{timestamp}_{file.filename}"
        file_location = os.path.join(UPLOAD_DIR, unique_filename)

        # Save the uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get the prediction from the model
        prediction = predict_image(file_location)

        # Optional: Remove the file after prediction to prevent clutter
        # os.remove(file_location)

        # Return the filename and prediction as a response
        return {"filename": unique_filename, "prediction": int(prediction)}

    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=f"File not found: {str(fnf_error)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

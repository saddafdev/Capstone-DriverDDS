from fastapi import FastAPI, File, UploadFile
from PIL import Image
import cv2 as cv
from io import BytesIO
import uvicorn
import numpy as np
#Uninstalling tensorflow-cpu-2.6.0
import numpy as np
import keras

model = keras.models.load_model("my_model")
app = FastAPI()

@app.get("/")
def get_status():
    return {"status": "API is live & working!"}

@app.get("/info")
def get_info():
    return {"endpoints": "Available endpoints are :",
            "/register": "to add a relative",
            "/recognize": "to search relative "}

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    predictions = 0
    frame = cv.resize(open_cv_image, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    output = cv.resize(frame,(180,180))
    output = np.expand_dims(output, 0)
    predictions = [model.predict(output)]
    label = np.argmax(predictions)
    return f'{label}'

if __name__ == "__main__":
    uvicorn.run(app, debug=True)

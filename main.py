# 1. Imports
from fastapi import FastAPI, UploadFile, File
import cv2
import shutil, os, time
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from sqlalchemy import create_engine, Column, Integer, String, Float, TIMESTAMP, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 2. Create app
app = FastAPI()
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("bin_model.pth"))
model.eval()

DATABASE_URL = "mysql+pymysql://root:root@localhost/bin_monitoring"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class BinData(Base):
    __tablename__ = "bin_data"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(255))
    status = Column(String(20))
    level = Column(Float)
    alert = Column(Text)
    created_at = Column(TIMESTAMP)

    # CREATE TABLE (AUTO)
    Base.metadata.create_all(bind=engine)

# Global variables
UPLOAD_FOLDER = "images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

bin_history = []

# Helper Functions 

def detect_bin_status(image_path):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    classes = ["EMPTY", "HALF", "FULL"]
    status = classes[predicted.item()]

    # 🔹 Add meaningful level
    if status == "EMPTY":
        level = 10
    elif status == "HALF":
        level = 50
    else:
        level = 95

    return status, level


def check_alert(status, level):

    if status == "FULL":
        return "🚨 ALERT: Bin is FULL! Immediate action required"

    elif status == "HALF":
        return "⚠️ Warning: Bin is reaching capacity"

    else:
        return None

# ADD CLEANUP FUNCTION HERE 👇
def cleanup_images(folder="images", max_files=20):

    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getctime
    )

    if len(files) > max_files:
        for f in files[:-max_files]:
            os.remove(f)


# API ROUTES

@app.post("/upload-image")
def upload_image(file: UploadFile = File(...)):

    filename = f"{int(time.time())}_{file.filename}"
    file_path = f"{UPLOAD_FOLDER}/{filename}"

    # Save image
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ML prediction
    status, level = detect_bin_status(file_path)

    # Alert
    alert = check_alert(status, level)

    # 🟢 SAVE TO DATABASE
    db = SessionLocal()

    new_record = BinData(
        image_path=file_path,
        status=status,
        level=level,
        alert=alert
    )

    db.add(new_record)
    db.commit()
    db.close()

    return {
        "status": status,
        "level": level,
        "alert": alert
    }

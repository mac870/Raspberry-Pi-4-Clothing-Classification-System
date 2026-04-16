from PIL import Image
import numpy as np
import time
from picamera2 import Picamera2
from cnn_mac import CNNmac

# Clothes type
clothing_types = [
    "T-shirt/top", "Pants", "Sweater", "Dress", "Jacket",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot"
]

def get_ready_image(pic_path):
    # Load and convert the picture to grayscale
    try:
        pic = Image.open(pic_path).convert("L")
        print(f"[INFO] Image '{pic_path}' loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        return None
    
    # Make it the right size for model
    pic = pic.resize((28, 28))
    
    # Convert to numbers between 0 and 1
    pic_data = np.array(pic) / 255.0
    
    # adjust shape
    return pic_data.reshape(28, 28, 1)

# clothing analyzer
smart_detector = CNNmac()
try:
    smart_detector.load("clothes_model.npz")
    print("[INFO] Model 'clothes_model.npz' loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# Set up the camera
my_camera = Picamera2()
my_camera.preview_configuration.main.size = (640, 480)
my_camera.preview_configuration.main.format = "RGB888"
my_camera.configure("preview")

# Let the camera warm up
my_camera.start()
time.sleep(2)

# Take a picture
snapshot_path = "camera_image.jpg"
my_camera.capture_file(snapshot_path)
print(f"Saved as {snapshot_path}")

# picture being classified
picture_data = get_ready_image(snapshot_path)
if picture_data is not None:
    result = smart_detector.forward(picture_data)
    best_guess = np.argmax(result)
    print("Clothes item:", clothing_types[best_guess])

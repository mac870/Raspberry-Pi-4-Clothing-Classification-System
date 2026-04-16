import numpy as np
from PIL import Image
from cnn_mac import CNNmac  

# Clothes type
clothing_types = [
    "T-shirt/top", "Pants", "Sweater", "Dress", "Jacket",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot"
]

def get_image_ready(pic_path):
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

    # Adjust shape 
    return pic_data.reshape(28, 28, 1)

# clothing analyzer
smart_detector = CNNmac()
try:
    smart_detector.load("clothes_model.npz")
    print("[INFO] Model 'clothes_model.npz' loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# picture being classified
picture_data = get_image_ready("test_image.jpg")  
if picture_data is not None:
    result = smart_detector.forward(picture_data)
    best_pred = np.argmax(result) #prediction
    print("Clothes item:", clothing_types[best_pred])

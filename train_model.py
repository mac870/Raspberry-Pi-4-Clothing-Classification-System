import numpy as np
from cnn_mac import CNNmac 

# Load training data from Fashion-MNIST
data = np.load("fashion_data.npz")
images, labels = data["X_train"], data["y_train"]

# Shuffle the data
indices = np.arange(len(images))
np.random.shuffle(indices)
images, labels = images[indices], labels[indices]

# Create model 
model = CNNmac()

# Train the model
model.train(
    images[:6000],    #  subset: first 1000 samples
    labels[:6000],    #  subset: corresponding labels
    epochs=10,         # train over 5 full cycles
    learn_rate=0.005  #  adjust weights gradually
)

# Save the trained model
model.save("clothes_model.npz")
print("Model saved as 'clothes_model.npz'")

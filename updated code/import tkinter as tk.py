import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Path to the pre-trained model
model_path = r'C:\Users\jayad\Downloads\updated code\pre trained.h5'  # Use your specific path here
model = load_model(model_path)

# Function to open file dialog and get image path
def get_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        preprocess_and_predict(file_path)

# Function to preprocess the image and predict the crowd count
def preprocess_and_predict(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Resize image to model's input size
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the crowd count
    prediction = model.predict(img_array)
    crowd_count = int(prediction[0][0])  # Convert the prediction to an integer
    print(f"Estimated number of people in the stadium: {crowd_count}")

    # Display the image and predicted count
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Crowd Count: {crowd_count}")
    plt.show()

# Set up the Tkinter window
root = tk.Tk()
root.title("Crowd Counting")

# Create a button to ask for image input
button = tk.Button(root, text="Select Image", command=get_image)
button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()

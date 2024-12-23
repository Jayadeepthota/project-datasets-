import tkinter as tk
from tkinter import filedialog
from tkinter import StringVar
from tkinter import messagebox
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image, ImageTk
import cv2
import torchvision.transforms as T
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load Faster R-CNN for crowd counting
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load models for age and gender detection
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]

face = cv2.dnn.readNet(face_pb, face_pbtxt)
age = cv2.dnn.readNet(age_model, age_prototxt)
gen = cv2.dnn.readNet(gender_model, gender_prototxt)

age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male', 'Female']

# Define true_labels and predicted_labels
true_labels = [1, 2, 3, 4, 5]  # Replace with your actual labels
predicted_labels = [1, 2, 0, 4, 5]  # Replace with your predictions

# Functions for crowd counting
def get_prediction(img_path, threshold=0.8):
    """Predict head count using Faster R-CNN."""
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_scores = list(pred[0]['scores'].detach().numpy())
    pred_classes = list(pred[0]['labels'].numpy())

    head_count = sum(1 for i, score in enumerate(pred_scores) if score > threshold and pred_classes[i] == 1)
    return head_count

def count_from_images():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image Files", "*.jpg;*.png"), ("All Files", "*.*")))
    if not file_path:
        return
    head_count = get_prediction(file_path)
    print(f"Total Head Count: {head_count}")
    image = cv2.imread(file_path)
    cv2.putText(image, f"Total Heads: {head_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Crowd Counting", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def count_from_video():
    file_path = filedialog.askopenfilename(title="Select a Video", filetypes=(("Video Files", "*.mp4;*.avi"), ("All Files", "*.*")))
    if not file_path:
        return
    video = cv2.VideoCapture(file_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        temp_file = "temp_frame.jpg"
        cv2.imwrite(temp_file, frame)
        head_count = get_prediction(temp_file)
        cv2.putText(frame, f"Total Heads: {head_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Crowd Counting from Video", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

# Functions for age and gender detection
def detect_age_gender():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image Files", "*.jpg;*.png"), ("All Files", "*.*")))
    if not file_path:
        return

    image = cv2.imread(file_path)
    image_resized = cv2.resize(image, (720, 640))
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)
    face.setInput(blob)
    detected_faces = face.forward()

    img_h, img_w = image_resized.shape[:2]
    if detected_faces.shape[2] == 0:
        print("No faces detected.")
        return

    for i in range(detected_faces.shape[2]):
        confidence = detected_faces[0, 0, i, 2]
        if confidence > 0.8:
            x1 = int(detected_faces[0, 0, i, 3] * img_w)
            y1 = int(detected_faces[0, 0, i, 4] * img_h)
            x2 = int(detected_faces[0, 0, i, 5] * img_w)
            y2 = int(detected_faces[0, 0, i, 6] * img_h)
            face_region = image_resized[max(0, y1):y2, max(0, x1):x2]
            blob = cv2.dnn.blobFromImage(face_region, 1.0, (227, 227), MODEL_MEAN_VALUES, True)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            # Predict gender
            gen.setInput(blob)
            gender = gender_classifications[gen.forward()[0].argmax()]

            # Predict age
            age.setInput(blob)
            age_range = age_classifications[age.forward()[0].argmax()]

            cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_resized, f"{gender}, {age_range}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Age and Gender Detection", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Functions to calculate and display metrics
def show_accuracy():
    global true_labels, predicted_labels
    accuracy = accuracy_score(true_labels, predicted_labels)
    messagebox.showinfo("Accuracy", f"Accuracy: {accuracy * 100:.2f}%")

def show_f1_score():
    global true_labels, predicted_labels
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    messagebox.showinfo("F1-Score", f"F1-Score: {f1:.2f}")

def show_error_rate():
    global true_labels, predicted_labels
    accuracy = accuracy_score(true_labels, predicted_labels)
    error_rate = 1 - accuracy
    messagebox.showinfo("Error Rate", f"Error Rate: {error_rate * 100:.2f}%")

# Function for example prediction
def example_prediction():
    # Sample data
    data = pd.DataFrame({
        "last_match_attendance": [10000, 15000, 13000, 20000, 17000, 25000, 22000],
        "weather": ["sunny", "cloudy", "sunny", "rainy", "sunny", "cloudy", "sunny"],
        "day_of_week": ["Monday", "Wednesday", "Friday", "Tuesday", "Thursday", "Friday", "Monday"],
        "current_match_attendance": [12000, 17000, 14000, 23000, 18000, 26000, 24000]
    })

    # Separating input features (X) and target (y)
    X = data[["last_match_attendance", "weather", "day_of_week"]]
    y = data["current_match_attendance"]

    # Encoding categorical features
    column_transformer = ColumnTransformer(
        transformers=[
            ("weather", OneHotEncoder(), ["weather"]),
            ("day_of_week", OneHotEncoder(), ["day_of_week"])
        ],
        remainder="passthrough"
    )

    X_transformed = column_transformer.fit_transform(X)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Create and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Example prediction
    example_input = pd.DataFrame({
        "last_match_attendance": [18000],
        "weather": ["rainy"],
        "day_of_week": ["Wednesday"]
    })
    example_input_transformed = column_transformer.transform(example_input)
    predicted_attendance = model.predict(example_input_transformed)

    # Feature importance visualization
    coefficients = model.coef_
    feature_names = (
        column_transformer.named_transformers_['weather'].get_feature_names_out(['weather']).tolist() +
        column_transformer.named_transformers_['day_of_week'].get_feature_names_out(['day_of_week']).tolist() +
        ["last_match_attendance"]
    )
    example_feature_impact = coefficients * example_input_transformed.toarray()[0]

    # Plotting
    plt.figure(figsize=(16, 6))
    plt.barh(feature_names, example_feature_impact, color='lightgreen')
    plt.xlabel("Feature Contribution to Predicted Attendance")
    plt.title("Feature Impact on Example Prediction")
    plt.show()

    messagebox.showinfo("Example Prediction", f"Predicted attendance: {predicted_attendance[0]:.2f}")

# Create the main Tkinter window
root = tk.Tk()
root.title("Crowd Counting and Age/Gender Detection")
root.geometry("800x600")

# Set the background with the UEL logo
bg_image = Image.open(r"C:\Users\jayad\Downloads\updated code\models\uel_logo.png.png")
bg_image = bg_image.resize((800, 600))  # Resize to match the window size
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# Create Frames for layout
title_frame = tk.Frame(root, bg="DarkGoldenrod1", bd=5, relief="solid")
title_frame.pack(fill="x", pady=10)

button_frame = tk.Frame(root, bd=5, relief="solid", bg="lightgray")
button_frame.pack(pady=20)

path_frame = tk.Frame(root, bd=5, relief="solid", bg="lightgray")
path_frame.pack(fill="x", pady=10)

# Title Section
font = ('times', 16, 'bold')
title = tk.Label(title_frame, text="Crowd Counting and Age/Gender Detection", font=font, bg="DarkGoldenrod1", fg="black")
title.pack(pady=10)

# Control Buttons Section
btn_count_img = tk.Button(button_frame, text="Count Heads in Image", command=count_from_images, font=('times', 12, 'bold'))
btn_count_img.grid(row=0, column=0, padx=10, pady=10)

btn_count_video = tk.Button(button_frame, text="Count Heads in Video", command=count_from_video, font=('times', 12, 'bold'))
btn_count_video.grid(row=1, column=0, padx=10, pady=10)

btn_age_gender = tk.Button(button_frame, text="Detect Age and Gender", command=detect_age_gender, font=('times', 12, 'bold'))
btn_age_gender.grid(row=2, column=0, padx=10, pady=10)

btn_accuracy = tk.Button(button_frame, text="Show Accuracy", command=show_accuracy, font=('times', 12, 'bold'))
btn_accuracy.grid(row=3, column=0, padx=10, pady=10)

btn_f1_score = tk.Button(button_frame, text="Show F1-Score", command=show_f1_score, font=('times', 12, 'bold'))
btn_f1_score.grid(row=4, column=0, padx=10, pady=10)

btn_error_rate = tk.Button(button_frame, text="Show Error Rate", command=show_error_rate, font=('times', 12, 'bold'))
btn_error_rate.grid(row=5, column=0, padx=10, pady=10)

# Adding empty button for example prediction
btn_example_prediction = tk.Button(button_frame, text="Match Attendance Prediction", command=example_prediction, font=('times', 12, 'bold'))
btn_example_prediction.grid(row=6, column=0, padx=10, pady=10)

# Path Display Section
path_var = StringVar()
path_label = tk.Label(path_frame, textvariable=path_var, font=('times', 12), anchor="w", justify="left")
path_label.pack(padx=10, pady=5)

root.mainloop()

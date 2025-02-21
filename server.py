# ## Step 3:
# ## Run the WebSocket Server
import cv2
import base64
import numpy as np
from flask import Flask
from flask_socketio import SocketIO
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Ensure 'best.pt' is in the same directory

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return "WebSocket Server Running"

@socketio.on("video_frame")
def handle_video_stream(data):
    try:
        # Decode base64 image
        img_data = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect objects
        results = model(frame)
        detected = False  # Flag to check if we should send the frame

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label = model.names[class_id]

                # Filter objects (Detect only relevant objects)
                if label in ["helmet", "BOX"] and confidence > 0.5:
                    detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Send frame only if an object was detected
        if detected:
            _, buffer = cv2.imencode(".jpg", frame)
            processed_frame = base64.b64encode(buffer).decode("utf-8")
            socketio.emit("processed_frame", f"data:image/jpeg;base64,{processed_frame}")

    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)



# ## Step 1:
# ## Download and Prepare the Dataset
# #!pip install roboflow
# # train: /var/www/html/object_detection/datasets/BOX-TRAIN-1/train/images


# from roboflow import Roboflow
# rf = Roboflow(api_key="IMEP4NZDI7g5p9u5nhwa")
# project = rf.workspace("ramesh-xt9fe").project("box-train-dwz99")
# version = project.version(4)
# dataset = version.download("yolov8")


# ## Step 2:                
# ## Train and Save the Model (Run Separately)
# from ultralytics import YOLO
# # Load YOLOv8 pre-trained model
# model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano (smallest, fastest)
# # Train the model on the dataset
# model.train(data="datasets/BOX-TRAIN-4/data.yaml", epochs=50, imgsz=640, batch=16)
# # Save the best model
# import shutil
# shutil.copy("runs/detect/train/weights/best.pt", "./best.pt")

# print("Training complete! 'best.pt' is saved in the current directory.")

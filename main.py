from ultralytics import YOLO
import cv2
import utils
from sort.sort import *
from utils import get_car, read_license_plate, write_csv

# Dictionary to store results
results = {}

# Initialize SORT tracker
mot_tracker = Sort()

# Load YOLO models for vehicle and license plate detection
coco_model = YOLO('yolov8n.pt')  # YOLO model for vehicle detection
license_plate_detector = YOLO('model_folder\weights/best.pt')  # YOLO model for license plate detection

# Open video file using OpenCV
cap = cv2.VideoCapture('./sample4.mp4')

# List of vehicle class IDs to consider
vehicles = [2, 3, 5, 7]

# Loop through video frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    # Check if the frame was successfully read
    if ret:
        # Break the loop after processing 10 frames (adjust as needed)
        if frame_nmr > 10:
            break
        
        # Initialize results dictionary for the current frame
        results[frame_nmr] = {}
        
        # Detect vehicles in the frame
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            # Check if the detected object is a vehicle
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track the detected vehicles using the SORT algorithm
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates in the frame
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to the corresponding vehicle
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # Check if a valid vehicle ID is assigned
            if car_id != -1:
                # Crop the license plate from the frame
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process the license plate image
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read the license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Check if the license plate text is successfully read
                if license_plate_text is not None:
                    # Store the results in the dictionary
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                          'text': license_plate_text,
                                          'bbox_score': score,
                                          'text_score': license_plate_text_score}
                    }

# Write the results to a CSV file
write_csv(results, './test.csv')

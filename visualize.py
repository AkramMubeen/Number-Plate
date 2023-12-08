import ast
import cv2
import numpy as np
import pandas as pd

# Function to draw a border around a region of interest in an image
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    # Extract coordinates of the top-left and bottom-right corners
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw lines to create a border around the region of interest
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Read results from a CSV file
results = pd.read_csv('./test_interpolated.csv')

# Open the video file using OpenCV
video_path = 'sample4.mp4'
cap = cv2.VideoCapture(video_path)

# Specify video codec and create a VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

# Dictionary to store license plate information for each car ID
license_plate = {}

# Loop through unique car IDs in the results
for car_id in np.unique(results['car_id']):
    # Find the row with the maximum license plate score for the current car ID
    max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    max_row = results[(results['car_id'] == car_id) & (results['license_number_score'] == max_score)].iloc[0]

    # Extract license plate information
    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': max_row['license_number']
    }

    # Set the video frame position to the one with the maximum license plate score
    cap.set(cv2.CAP_PROP_POS_FRAMES, max_row['frame_nmr'])
    ret, frame = cap.read()

    # Extract license plate bounding box coordinates and crop the license plate
    x1, y1, x2, y2 = ast.literal_eval(max_row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    # Store the cropped license plate in the dictionary
    license_plate[car_id]['license_crop'] = license_crop

# Initialize frame number
frame_nmr = -1

# Set the video frame position to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Read frames from the video
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        # Get rows in the results DataFrame corresponding to the current frame number
        df_ = results[results['frame_nmr'] == frame_nmr]

        # Iterate over each row in the DataFrame for the current frame
        for row_indx in range(len(df_)):
            # Draw a border around the car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Draw a rectangle around the license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Crop and overlay the license plate on the car
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
            H, W, _ = license_crop.shape

            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop
                frame[int(car_y1) - H - 400:int(car_y1) - H - 100, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                # Draw the license plate number on the frame
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)
            except:
                pass

        # Write the frame to the output video
        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # Display the frame (commented

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

out.release()
cap.release()

import cv2
import cv2.aruco as aruco
import os

# Choose dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# Output folder
output_dir = "/home/newin/Projects/gazebo_models/aruco_new"
os.makedirs(output_dir, exist_ok=True)

# Generate 20 markers (IDs 0–19)
for i in range(20):
    marker_img = aruco.drawMarker(aruco_dict, i, 200)  # 200 = size in pixels
    file_path = os.path.join(output_dir, f"Marker{i}.png")
    cv2.imwrite(file_path, marker_img)
    print(f"Saved {file_path}")

import cv2
import cv2.aruco as aruco
import os

# Output folder
os.makedirs("aruco_markers", exist_ok=True)

# Choose dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

# Number of markers you want
num_markers = 30
marker_size = 200  # pixels

for marker_id in range(num_markers):
    img = aruco.drawMarker(aruco_dict, marker_id, marker_size)
    filename = f"aruco_markers/aruco_marker_{marker_id}.png"
    cv2.imwrite(filename, img)

print(f"✅ Generated {num_markers} ArUco markers in 'aruco_markers/' folder")

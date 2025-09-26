import cv2
import numpy as np
import os
import glob

images_folder = "checkerboard"   # your folder with photos
output_folder = "annotated"      # folder to save annotated images
pattern_size = (15, 10)            # (columns, rows) of INTERNAL corners
square_size = 1.0                # can keep 1.0 if you don't know the size

# Get the current script directory and create full paths
script_dir = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(script_dir, images_folder)
output_folder = os.path.join(script_dir, output_folder)
os.makedirs(output_folder, exist_ok=True)

# Prepare object points (3D real world coords)
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points

# Find images
images = glob.glob(os.path.join(images_folder, "*.jpg")) + \
         glob.glob(os.path.join(images_folder, "*.png"))

print(f"Looking for images in: {images_folder}")
print(f"Saving annotated images to: {output_folder}")
print(f"Images found: {len(images)}")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print("Could not read:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try the standard method first with flags
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    # If not found, try the robust SB method
    if not found:
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size)

    if found:
        # Refine corners 
        if corners is not None and corners.shape[0] == pattern_size[0]*pattern_size[1]:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and save annotated image
        annotated = img.copy()
        cv2.drawChessboardCorners(annotated, pattern_size, corners, found)
        outname = os.path.join(output_folder, os.path.basename(fname))
        cv2.imwrite(outname, annotated)
        print("✔ Corners found in", os.path.basename(fname))
    else:
        print("✘ No corners in", os.path.basename(fname))
        # Optional: show image for debugging
        # cv2.imshow("Failed", img)
        # cv2.waitKey(0)

# Run calibration if enough images worked
if len(objpoints) > 0:
    h, w = gray.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    np.savez("camera_params.npz", camera_matrix=mtx, dist_coefs=dist)
    print("Calibration done. Parameters saved to camera_params.npz")
else:
    print("No valid images found!")


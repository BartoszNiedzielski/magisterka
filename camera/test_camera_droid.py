import cv2
import numpy as np
import os

# CONFIGURATION
CAMERA_INDEX = 2
MODEL_RES = (224, 224)  # Standard for Pi0 / Droid models

def verify_perspective():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    print("--- Perspective Verification Mode ---")
    print("1. Ensure the Robot Base is in the bottom 1/3rd of the frame.")
    print("2. Ensure the full workspace is visible.")
    print("3. Press 's' to save a test frame, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. Square Crop (Models usually expect square input)
            h, w, _ = frame.shape
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            crop = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
            
            # 2. Resize to Model Input
            input_view = cv2.resize(crop, MODEL_RES)

            # 3. Draw DROID-style alignment guides
            # Draw a circle where the Robot Base center should ideally sit
            cv2.circle(input_view, (112, 180), 10, (0, 255, 255), -1) 
            cv2.putText(input_view, "ROBOT BASE HERE", (60, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Show the view
            cv2.imshow("Pi0 Zero-Shot Preview", input_view)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("pi0_perspective_test.jpg", input_view)
                print("Captured 'pi0_perspective_test.jpg' for review.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_perspective()
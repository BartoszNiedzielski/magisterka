import cv2
import time

# === CONFIGURATION ===
# 0 is usually the default laptop webcam or the first USB camera plugged in.
# If you still have the RealSense plugged in, the Logitech might be 1, 2, or 3.
CAMERA_INDEX = 0  

def test_camera():
    print(f"[*] Attempting to open camera at index {CAMERA_INDEX}...")
    
    # Initialize the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"[!] ERROR: Could not open camera {CAMERA_INDEX}.")
        print("[!] Try changing CAMERA_INDEX to 1, 2, or 3.")
        return

    # Get the default resolution of the camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[*] Camera Online! Resolution: {width}x{height} @ {fps} FPS")
    print("[*] Press 'q' to quit.")

    # Frame rate calculation variables
    prev_frame_time = 0
    new_frame_time = 0

    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            if not ret:
                print("[!] ERROR: Failed to grab a frame from the camera.")
                break
                
            # Calculate actual FPS
            new_frame_time = time.time()
            actual_fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Draw the FPS on the screen so you can check for lag
            cv2.putText(frame, f"FPS: {int(actual_fps)}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the video feed
            cv2.imshow("Logitech Camera Test", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Always cleanly release the camera and destroy windows
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Camera released and shut down.")

if __name__ == "__main__":
    test_camera()
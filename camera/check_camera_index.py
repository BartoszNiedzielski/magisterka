import cv2

print("Scanning for cameras...")
for index in range(10):
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        print(f"✅ Camera found at index: {index}")
        cap.release()
    else:
        cap.release()
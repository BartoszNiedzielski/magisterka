import pyrealsense2 as rs
import numpy as np
import cv2

# 1. Configure the stream
pipeline = rs.pipeline()
config = rs.config()

# Enable the streams you want to use
# Resolution: 640x480, FPS: 30
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 2. Start the pipeline
pipeline.start(config)

try:
    while True:
        # 3. Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # 4. Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 5. Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #    This makes the depth image easier to visualize
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # 6. Show images
        cv2.imshow('RealSense', images)
        
        # Press 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 7. Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
import cv2
import numpy as np

# Define the range for detecting the color of the sign (e.g., red)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

def detect_sign(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour is the sign
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw the bounding box and centroid
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
        
        return cx, cy, w * h
    
    return None, None, None

def main():
    video_path = 'path_to_your_video_file.mp4'
    cap = cv2.VideoCapture(0)  # Use 0 instead of video_path for webcam

    kP = 0.1  # Proportional gain for control

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the sign
        cx, cy, area = detect_sign(frame)
        
        if cx is not None and cy is not None:
            # Frame center
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            # Calculate error between the sign position and the frame center
            error_x = cx - frame_center_x
            error_y = cy - frame_center_y

            # Calculate control signals
            control_signal_x = kP * error_x
            control_signal_y = kP * error_y

            # Simulate sending control signals to motors
            print(f"Control Signal X: {control_signal_x}, Control Signal Y: {control_signal_y}")

            # Simulate robot movement based on control signals
            if control_signal_x > 0:
                print(f"Turn right by {control_signal_x} units")
            elif control_signal_x < 0:
                print(f"Turn left by {-control_signal_x} units")
            
            if control_signal_y > 0:
                print(f"Move forward by {control_signal_y} units")
            elif control_signal_y < 0:
                print(f"Move backward by {-control_signal_y} units")
        
        # Display the frame with detected sign
        cv2.imshow('Frame with Sign Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

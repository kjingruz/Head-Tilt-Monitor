import cv2
import dlib
import numpy as np

# Define the draw function to draw the axis
def draw(img, imgpts, imgpts_axis):
    corner = tuple(imgpts[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts_axis[0].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts_axis[1].ravel().astype(int)), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts_axis[2].ravel().astype(int)), (0,0,255), 5)
    return img

def rotation_vector_to_euler_angles(rotation_vector):
    # Calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    
    # Transformed to quaterniond
    w = np.cos(theta / 2)
    x = np.sin(theta / 2)*rotation_vector[0][0] / theta
    y = np.sin(theta / 2)*rotation_vector[1][0] / theta
    z = np.sin(theta / 2)*rotation_vector[2][0] / theta
    
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Convert to degrees
    roll_x = np.degrees(roll_x)
    pitch_y = np.degrees(pitch_y)
    yaw_z = np.degrees(yaw_z)
    
    return roll_x, pitch_y, yaw_z

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Load the facial landmarks predictor from dlib
# You need to have the 'shape_predictor_68_face_landmarks.dat' file in your working directory or specify the path to it.
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)

# Define the 3D model points (these would be based on an average head and should be adjusted according to your specific use case):
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right Mouth corner
])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Grab a frame to get the width and height
ret, frame = cap.read()
if ret:
    frame_height, frame_width = frame.shape[:2]
else:
    print("Failed to grab frame")
    cap.release()
    exit()

# Define the camera matrix based on the frame size
focal_length = frame_width
center = (frame_width // 2, frame_height // 2)
camera_matrix = np.array(
                 [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
             )

# Assume no lens distortion
dist_coeffs = np.zeros((4,1))

# Get the camera matrix (you'll need to calibrate your camera to get these values)
size = (frame_width, frame_height)
focal_length = size[1]
center = (size[1]//2, size[0]//2)
camera_matrix = np.array(
                 [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype = "double"
             )

# Assume no lens distortion
dist_coeffs = np.zeros((4,1))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_detector(gray, 1)
    
    for detection in detections:
        landmarks = face_predictor(gray, detection)
        
        # Visualize the landmarks.
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        # Define the 2D image points of the detected facial landmarks (You need to use the correct points here)
        image_points = np.array([
            (landmarks.part(33).x, landmarks.part(33).y),     # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),       # Chin
            (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),     # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)      # Right Mouth corner
        ], dtype="double")

        # Solve for the pose
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        # Calculate Euler angles from the rotation vector
        roll, pitch, yaw = rotation_vector_to_euler_angles(rotation_vector)

        # Display the angles on the screen
        cv2.putText(frame, "Roll: {:.2f}".format(roll), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Pitch: {:.2f}".format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Yaw: {:.2f}".format(yaw), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check for "dangerous" angles and display an alert
        if abs(roll) > 30 or abs(pitch) > 30 or abs(yaw) > 30:  # These threshold values can be adjusted
            cv2.putText(frame, "DANGEROUS ANGLE!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Project a 3D axis to visualize the pose
        # This function will create a few 3D points to draw the axis.
        axis = np.float32([[500, 0, 0], [0, 500, 0], [0, 0, 500]])
        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        frame = draw(frame, image_points, imgpts)

    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

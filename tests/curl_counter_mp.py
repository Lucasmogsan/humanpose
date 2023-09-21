import cv2
import argparse
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 test')
    parser.add_argument("--video-scale", default=1, type=float, help="Scale factor for video")
    parser.add_argument('--src', default='muscle_up_test_short.mov', type=str, help='Path to video file')
    parser.add_argument('--vis', action='store_true', default=False, help='Visualize processed images')
    parser.add_argument('--save-to', default='None', type=str, help='Save output to file')
    args = parser.parse_args()
    return args


def get_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:    # Maximum angle is 180for joints
        angle = 360-angle

    return angle

def main():
    # Parse arguments
    args = parse_args()
    video_path = args.src

    counter = 0
    stage = None

    # Load video
    if video_path == '0' or video_path == '1': # Webcam
        cap = cv2.VideoCapture(int(video_path))
    else: # Video file
        cap = cv2.VideoCapture(video_path)

    _, frame = cap.read()
    new_width = int(frame.shape[1] * args.video_scale)
    new_height = int(frame.shape[0] * args.video_scale)
    new_dim = (new_width, new_height)



    if args.save_to != 'None':
        out = cv2.VideoWriter(f'runs/{args.save_to}_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 8, new_dim)

    # Setup mediapipe instance:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Loop through frames
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, new_dim, interpolation = cv2.INTER_AREA)

                # Recolor image to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe needs RGB
                frame.flags.writeable = False
                # Make detection
                results = pose.process(frame)   # Store detections in result
                # Recolor back to BGR
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                except:
                    pass
                
                # Get coordinates of joints
                r_shoulder_xy = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow_xy = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist_xy = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # Calculate angle between joints
                angle = get_angle(r_shoulder_xy, r_elbow_xy, r_wrist_xy)

                # Curl counter
                if angle > 150:
                    stage = "down"
                if angle < 30 and stage =='down':
                    stage="up"
                    counter += 1
            
                # Display
                if args.vis:
                    # Render detections
                    if angle > 150:
                        draw_color = (255,0,0)
                    elif angle < 30:
                        draw_color = (0,255,0)
                    else:
                        draw_color = (150,150,150)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245,177,66), thickness=2, circle_radius=2), # landmark drawing spec
                                              mp_drawing.DrawingSpec(color=draw_color, thickness=2, circle_radius=2)  # connection drawing spec
                                              )
                    
                    
                    # Display angle
                    cv2.putText(frame, str(angle),
                                tuple(np.multiply(r_elbow_xy, new_dim).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Display rep counter
                    cv2.rectangle(frame, (0,0), (150,70), (0,0,0), -1)
                    cv2.putText(frame, 'REPS', (15, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(frame, str(counter), (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (150,150,150), 2, cv2.LINE_AA)


                    cv2.imshow('preview', frame)

                # write the frame
                if args.save_to != 'None':
                    out.write(frame)

                # Break if q is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break   # end of video

            if cv2.waitKey(30) == ord('q'): # press q to quit
                break

        # Release everything if job is finished
        if args.save_to != 'None':
            out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
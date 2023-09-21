import cv2
import argparse
from ultralytics import YOLO
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 test')
    parser.add_argument("--video-scale", default=1, type=float, help="Scale factor for video")
    parser.add_argument('--video-path', default='muscle_up_test_short.mov', type=str, help='Path to video file')
    parser.add_argument('--vis', action='store_true', default=False, help='Visualize processed images')
    parser.add_argument('--save-to', default='None', type=str, help='Save output to file')
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()
    video_path = args.video_path

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

    # Load yolov8 pose model
    model = YOLO('yolov8m-pose.pt')

    # Loop through frames
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.resize(frame, new_dim, interpolation = cv2.INTER_AREA)

            # Inference from YOLO model
            results = model(frame, conf=0.5)
            #print(results)
            # Visualize
            annotated_frame = results[0].plot()
            #print(annotated_frame)

            # Display
            if args.vis:
                cv2.imshow('Inference', annotated_frame)
                #cv2.imshow('preview', frame)

            # write the frame
            if args.save_to != 'None':
                out.write(annotated_frame)

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
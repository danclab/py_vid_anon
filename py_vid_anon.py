import sys

import cv2
import os
from mtcnn import MTCNN

def blur_faces_in_frame(frame, detector=None, confidence_threshold=0.90):
    """blur_faces_in_frame
    Detects faces in 'frame' using MTCNN and blurs them in place.

    :param frame: A BGR image (as read by OpenCV).
    :param detector: An instance of MTCNN. If None, a new one will be created.
    :param confidence_threshold: Minimum confidence score for a detection to be considered valid.
    :return: The modified frame with blurred faces.
    """
    # If no detector is provided, create a new one (not ideal for high-performance loops,
    # since creating the detector is expensive)
    if detector is None:
        detector = MTCNN()

    # MTCNN expects an RGB image, so convert from BGR (OpenCV default) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    faces = detector.detect_faces(frame_rgb)

    # For each face, blur the region in the original BGR frame
    for face in faces:
        confidence = face.get('confidence', 0)
        if confidence >= confidence_threshold:
            # face['box'] is [x, y, width, height]
            x, y, w, h = face['box']

            # Ensure bounding box is within the frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            # Extract region of interest (ROI) for the face
            roi = frame[y:y+h, x:x+w]
            # Apply a strong blur
            roi_blurred = cv2.GaussianBlur(roi, (61, 61), 0)
            # Place it back into the frame
            frame[y:y+h, x:x+w] = roi_blurred

    return frame

def anonymize_video(
    video_file,
    confidence_threshold=0.90
):
    """
    Anonymizes (blurs faces in) the given video file.

    :param video_file: Path to the input video file.
    :param face_cascade_path: Path to the Haar cascade for face detection.
    :param scale_factor: Parameter specifying how much the image size is reduced
                         at each image scale in face detection.
    :param min_neighbors: Parameter specifying how many neighbors each candidate
                          rectangle should have to retain it during face detection.
    """
    # Load the Haar cascade
    # face_cascade_path=os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier(face_cascade_path)
    # if face_cascade.empty():
    #     raise IOError(f"Could not load face cascade from {face_cascade_path}")
    # Create a single MTCNN detector (more efficient than creating for each frame)
    detector = MTCNN()

    # Parse filename and extension to create output name
    filename, ext = os.path.splitext(video_file)
    out_file = f"{filename}_anon{ext}"

    # Open the video capture
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_file}")

    # Gather video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Blur faces in this frame using MTCNN
        frame_blurred = blur_faces_in_frame(
            frame=frame,
            detector=detector,
            confidence_threshold=confidence_threshold
        )

        # Write the blurred frame to the output video
        out.write(frame_blurred)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Anonymized video saved as: {out_file}")


if __name__=='__main__':
    fname=sys.argv[1]
    anonymize_video(fname)

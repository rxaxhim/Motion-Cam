import cv2
from datetime import datetime
import os
import pygame.mixer

# Initialize pygame mixer
pygame.mixer.init()

# Ensure there's a directory to save video clips
if not os.path.exists("motion_clips"):
    os.makedirs("motion_clips")

# Initialize camera and parameters
camera = cv2.VideoCapture(0)
prev_frame = None
recording = False
video_writer = None
alert_duration = 10  # Duration (in seconds) to record after detecting motion
frames_recorded = 0

while True:
    ret, frame = camera.read()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    frame_diff = cv2.absdiff(prev_frame, gray)
    threshold = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold, None, iterations=2)
    (contours, _) = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    if motion_detected:
        cv2.putText(frame, "Status: Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Start recording if not already doing so
        if not recording:
            filename = os.path.join("motion_clips", f"motion_{timestamp.replace(' ', '_').replace(':', '-')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
            pygame.mixer.music.load('beep.wav')
            pygame.mixer.music.play()

    else:
        cv2.putText(frame, "Status: No Motion", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame if recording
    if recording:
        video_writer.write(frame)
        frames_recorded += 1
        if frames_recorded >= alert_duration * 20:  # Assuming 20 FPS
            video_writer.release()
            recording = False
            frames_recorded = 0

    cv2.imshow("Motion Detection", frame)
    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# Release the video writer if it's still active
if recording:
    video_writer.release()

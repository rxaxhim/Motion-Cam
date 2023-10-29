from flask import Flask, render_template, Response, url_for, send_from_directory
import cv2
import os
from datetime import datetime
import pygame

app = Flask(__name__)
camera = cv2.VideoCapture(0)
pygame.mixer.init()

prev_frame = None
recording = False
video_writer = None
alert_duration = 10  # Duration (in seconds) to record after detecting motion
frames_recorded = 0

def generate_frames():
    global prev_frame, recording, video_writer, frames_recorded
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

        # By default, show "Status: No Motion"
        cv2.putText(frame, "Status: No Motion", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if motion_detected:
            # Clear the previous "No Motion" status and show "Motion Detected"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0,0,0), -1)
            cv2.putText(frame, "Status: Motion Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Start recording if not already doing so
            if not recording:
                filename = os.path.join("motion_clips", f"motion_{timestamp.replace(' ', '_').replace(':', '-')}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                recording = True
                pygame.mixer.music.load('beep.wav')
                pygame.mixer.music.play()

        # Save the frame if recording
        if recording:
            video_writer.write(frame)
            frames_recorded += 1
            if frames_recorded >= alert_duration * 20:  # Assuming 20 FPS
                video_writer.release()
                recording = False
                frames_recorded = 0

        prev_frame = gray
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    recordings = os.listdir("motion_clips")
    return render_template('index.html', recordings=recordings)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/motion_clips/<filename>')
def serve_recording(filename):
    return send_from_directory('motion_clips', filename)

if __name__ == '__main__':
    if not os.path.exists("motion_clips"):
        os.makedirs("motion_clips")
    app.run(host='0.0.0.0', port=5001, debug=True)
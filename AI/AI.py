import pygame
import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO
import requests
import time

# --- Config ---
STREAM_URL = 'http://192.168.1.137:81/stream'  # Change to your ESP32-CAM IP
ESP32_ACTION_URL = 'http://192.168.1.137/action?go='
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
CONFIDENCE_THRESHOLD = 0.6

# --- Init ---
pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("ESP32-CAM AI Tracker")
font = pygame.font.SysFont('Arial', 16)

# Load YOLO model
model = YOLO('yolov8n.pt')  # Make sure this model is downloaded

# MJPEG stream reader
def mjpeg_stream_reader(url):
    stream = urllib.request.urlopen(url)
    bytes_ = b''
    while True:
        bytes_ += stream.read(1024)
        a = bytes_.find(b'\xff\xd8')  # JPEG start
        b = bytes_.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_[a:b+2]
            bytes_ = bytes_[b+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield img

# Start stream generator
frames = mjpeg_stream_reader(STREAM_URL)
clock = pygame.time.Clock()
running = True
fps_display = 0
frame_counter = 0
start_time = time.time()

# --- Main Loop ---
while running:
    try:
        frame = next(frames)
    except Exception as e:
        print("Stream error:", e)
        continue

    # Resize to fit window
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Detect objects
    results = model(frame, verbose=False)[0]
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = box
        if conf < CONFIDENCE_THRESHOLD:
            continue
        label = f"{model.names[int(cls_id)]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # FPS counter
    frame_counter += 1
    if frame_counter >= 10:
        elapsed = time.time() - start_time
        fps_display = frame_counter / elapsed
        frame_counter = 0
        start_time = time.time()

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))

    window.blit(surface, (0, 0))

    # UI overlay
    pygame.draw.rect(window, (0, 0, 0), (0, 0, 260, 30))
    window.blit(font.render("W/A/S/D = PTZ | P = Snapshot | Q = Quit", True, (255, 255, 255)), (5, 5))
    fps_text = font.render(f"FPS: {fps_display:.1f}", True, (0, 255, 255))
    fps_rect = fps_text.get_rect(topright=(WINDOW_WIDTH - 10, 5))
    window.blit(fps_text, fps_rect)


    pygame.display.update()
    clock.tick(30)

    # Input handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_w:
                requests.get(ESP32_ACTION_URL + 'up')
            elif event.key == pygame.K_s:
                requests.get(ESP32_ACTION_URL + 'down')
            elif event.key == pygame.K_a:
                requests.get(ESP32_ACTION_URL + 'left')
            elif event.key == pygame.K_d:
                requests.get(ESP32_ACTION_URL + 'right')
            elif event.key == pygame.K_p:
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved: {filename}")

# --- Cleanup ---
pygame.quit()

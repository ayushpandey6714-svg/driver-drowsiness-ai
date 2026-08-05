import cv2
import mediapipe as mp
import math
import pygame
import os

pygame.mixer.init()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

sleep_counter = 0
no_face_counter = 0
SLEEP_THRESHOLD = 20    # Frames for eye closure
NO_FACE_THRESHOLD = 15  # Frames if face completely drops/disappears

alarm_sound = None
if os.path.exists("alarm.wav"):
    alarm_sound = pygame.mixer.Sound("alarm.wav")

print("Drowsiness & Head-Drop System is running... (Press 'q' to exit)")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from camera.")
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    eyes_closed = False
    face_detected = False

    if results.multi_face_landmarks:
        face_detected = True
        for face_landmarks in results.multi_face_landmarks:
            r_top = (int(face_landmarks.landmark[159].x * w), int(face_landmarks.landmark[159].y * h))
            r_bottom = (int(face_landmarks.landmark[145].x * w), int(face_landmarks.landmark[145].y * h))
            l_top = (int(face_landmarks.landmark[386].x * w), int(face_landmarks.landmark[386].y * h))
            l_bottom = (int(face_landmarks.landmark[374].x * w), int(face_landmarks.landmark[374].y * h))

            right_eye_dist = get_distance(r_top, r_bottom)
            left_eye_dist = get_distance(l_top, l_bottom)

            if right_eye_dist < 10 and left_eye_dist < 10:
                eyes_closed = True

            cv2.circle(frame, r_top, 3, (0, 255, 0), -1)
            cv2.circle(frame, r_bottom, 3, (0, 255, 0), -1)
            cv2.circle(frame, l_top, 3, (0, 255, 0), -1)
            cv2.circle(frame, l_bottom, 3, (0, 255, 0), -1)

    # Logic counters
    if eyes_closed:
        sleep_counter += 1
    else:
        sleep_counter = 0

    if not face_detected:
        no_face_counter += 1
    else:
        no_face_counter = 0

    # Trigger Alarm if eyes closed or face drops/disappears from frame
    if sleep_counter >= SLEEP_THRESHOLD or no_face_counter >= NO_FACE_THRESHOLD:
        if sleep_counter >= SLEEP_THRESHOLD:
            cv2.putText(frame, "ALERT: EYES CLOSED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "ALERT: HEAD DROPPED / NO FACE!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            
        if alarm_sound and not pygame.mixer.get_busy():
            alarm_sound.play(-1)
    else:
        if alarm_sound:
            alarm_sound.stop()
        cv2.putText(frame, "Status: Safe & Active", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Advanced Driver Drowsiness System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if alarm_sound:
            alarm_sound.stop()
        break

cap.release()
cv2.destroyAllWindows()
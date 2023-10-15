import cv2
import pygame
import time

face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
audio_playing = False  
start_time = None  

pygame.init()

alarm_sound = pygame.mixer.Sound('alarm.mp3')

while True:
    ret, frame = cap.read() 
    if not ret:
        continue 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade1.detectMultiScale(gray, 1.1, 6) # El parametro 1.1 cuanto mas grande sea el numero como maximo 1.9 detecta caras mas pequeñas.Y el aprametro 6 dice el numero de detecciones son para que de como valida la cara cuanto mas grande sea este numero menos falsos positivos pero tambien mas dificil la detecion
    
    # Check if faces are detected
    if len(faces) > 0:
        if not audio_playing:
            start_time = time.time()  
            audio_playing = True
        else:
            current_time = time.time()
            if current_time - start_time >= 0.5:  
                alarm_sound.play()
    else:
        if audio_playing:
            alarm_sound.stop()
            audio_playing = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

    cv2.imshow('Face Detection', frame) 
    
    k = cv2.waitKey(10)
    if k == 27:
        break

# Release pygame resources
pygame.quit()
cap.release()
cv2.destroyAllWindows()

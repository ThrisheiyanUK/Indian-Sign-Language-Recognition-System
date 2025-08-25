import cv2
import numpy as np
import mediapipe as mp
import os
import time

CLASS_NAMES = [chr(c) for c in range(ord('A'), ord('Z')+1)] + \
              [str(n) for n in range(10)] + ["space", "delete"]

DATA_DIR = './data_images'
KEYPOINTS_DIR = './data_keypoints'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KEYPOINTS_DIR, exist_ok=True)

dataset_size = 500  

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

for class_name in CLASS_NAMES:
    class_image_dir = os.path.join(DATA_DIR, class_name)
    class_kp_dir = os.path.join(KEYPOINTS_DIR, class_name)
    os.makedirs(class_image_dir, exist_ok=True)
    os.makedirs(class_kp_dir, exist_ok=True)

    print(f"\nCollecting data for class '{class_name}'")
    print("Press 'Q' when ready to start capturing, or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.putText(frame, f"Ready? Class: {class_name} (Press Q)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print("Starting capture in 2 seconds...")
    time.sleep(2)

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            x_all, y_all = [], []
            all_keypoints = []

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    x_all.append(lm.x * w)
                    y_all.append(lm.y * h)
                    all_keypoints.extend([lm.x, lm.y, lm.z])

            x_min = max(int(min(x_all)) - 20, 0)
            y_min = max(int(min(y_all)) - 20, 0)
            x_max = min(int(max(x_all)) + 20, w)
            y_max = min(int(max(y_all)) + 20, h)

            cropped_image = frame[y_min:y_max, x_min:x_max]
            
            # Resize to 224x224
            resized_image = cv2.resize(cropped_image, (224, 224))

            img_path = os.path.join(class_image_dir, f"{counter:03d}.jpg")
            cv2.imwrite(img_path, resized_image)

            kp_path = os.path.join(class_kp_dir, f"{counter:03d}.npy")
            np.save(kp_path, np.array(all_keypoints))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} - {counter+1}/{dataset_size}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            counter += 1
            time.sleep(0.15)

        else:
            cv2.putText(frame, "No hand detected", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("\n✅ Data collection completed!")
cap.release()
cv2.destroyAllWindows()
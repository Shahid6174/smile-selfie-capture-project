import cv2
import os
import time

# Initialize video capture
video = cv2.VideoCapture(0)

# Load Haar cascade classifiers
faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

# Counter for saving images
cnt = 500
save_path = r"D:\smile-selfie-capture-project\images"

# Ensure save directory exists
os.makedirs(save_path, exist_ok=True)

smile_start_time = None  # Track when the smile starts
hold_duration = 3  # Time in seconds for holding the smile

while True:
    success, img = video.read()
    if not success:
        print("Failed to capture video frame")
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)

    smile_detected = False  # Track if a smile is detected in this frame

    for (x, y, w, h) in faces:
        # Detect smiles within the detected face region
        roi_gray = grayImg[y:y+h, x:x+w]
        smiles = smileCascade.detectMultiScale(roi_gray, 1.8, 15)

        if len(smiles) > 0:
            smile_detected = True  # A smile is detected in this frame

            if smile_start_time is None:
                smile_start_time = time.time()  # Start the timer

            elapsed_time = time.time() - smile_start_time

            if elapsed_time >= hold_duration:
                # Save the image after 3 seconds of continuous smiling
                img_path = os.path.join(save_path, f"image_{cnt}.jpg")
                cv2.imwrite(img_path, img)
                print(f"Image {cnt} saved at {img_path}")

                # Pause for 2 seconds before exiting
                time.sleep(2)
                video.release()
                cv2.destroyAllWindows()
                exit()  # Stop the script after capturing the image

        else:
            smile_start_time = None  # Reset the timer if smile disappears

    # Display live video feed
    cv2.imshow('Live Video', img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

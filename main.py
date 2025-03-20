import cv2
import os
import time
import random

# Initialize video capture
video = cv2.VideoCapture(0)

# Load Haar cascade classifiers
faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

# Generate a random 6-digit folder name
folder_code = str(random.randint(100000, 999999))
folder_path = os.path.join(r"D:\smile-selfie-capture-project\images", folder_code)

# Ensure the new folder exists
os.makedirs(folder_path, exist_ok=True)

# Track number of images taken
images_captured = 0  
total_images = 3  

smile_start_time = None  # Track when the smile starts
hold_duration = 3  # Time in seconds for holding the smile

print(f"New session started. Images will be saved in: {folder_path}")

while images_captured < total_images:
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
                img_path = os.path.join(folder_path, f"image_{images_captured+1}.jpg")
                cv2.imwrite(img_path, img)
                print(f"Image {images_captured+1} saved at {img_path}")

                # Increment counter
                images_captured += 1

                # Reset the timer for next image
                smile_start_time = None

                # Wait before capturing the next image
                if images_captured < total_images:
                    print("Keep smiling! Next capture in 3 seconds...")
                    time.sleep(3)  # Ensures smile is held again before the next capture

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

import cv2 
import numpy as np



import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# initialize a face landmark detector
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# define a video capture object 
vid0 = cv2.VideoCapture(0)
vid1 = cv2.VideoCapture(1) 


# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.png")

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)

# grab frames multiple times util the capture buffer is empty
for i in range(10):
    vid0.grab()
    vid1.grab()

while(True): 
    # Capture the video frame by frame 
    vid0.grab()
    vid1.grab()
    _, frame0 = vid0.retrieve() 
    _, frame1 = vid1.retrieve()
  
    # Display the resulting frame 
    cv2.imshow('frame', np.concatenate((frame0, frame1), axis=1))
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid0.release()
vid1.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


# Create an FaceLandmarker object.
# !wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
base_options = mp.tasks.BaseOptions(model_asset_path=r'C:\Users\t-ruiwang\Documents\projects\live_mocap\mediapipe-assets\face_landmarker_v2_with_blendshapes.task')
options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       running_mode=mp.tasks.vision.RunningMode.IMAGE,
                                       num_faces=1)
detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

vid = cv2.VideoCapture(0)

while(True): 
    # Capture the video frame by frame 
    vid.grab()
    _, frame = vid.retrieve() 
  
    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
#    print(detection_result.face_blendshapes[0])

    # STEP 5: Process the detection result. In this case, visualize it.
    frame = draw_landmarks_on_image(frame, detection_result)

    # Display the resulting frame 
    cv2.imshow('frame', frame)
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release()
detector.close()
cv2.destroyAllWindows() 
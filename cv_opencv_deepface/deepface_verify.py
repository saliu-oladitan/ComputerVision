import json

from deepface import DeepFace


models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
  "Buffalo_L"
]


backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yolov11s',
  'yolov11n',
  'yolov11m',
  'yunet',
  'centerface',
]


alignment_modes = [True, False]

result = DeepFace.verify(
  img1_path = "cv_opencv_deepface/cv_images/saliu/saliu_img1.jpg",
  img2_path = "cv_opencv_deepface/cv_images/saliu/saliu_img2.jpg",
  model_name = models[2],
  enforce_detection=False # Without this, it throws an error if a face isn't detected
)

print(" ")
print(" ")
print(json.dumps(result, indent=2))


  # detector_backend = backends[0],
  # align = alignment_modes[0],
  # anti_spoofing = True
  # Real Time Analysis
  # API



# # anti spoofing test in face detection
# face_objs = DeepFace.extract_faces(
#   img_path="dataset/img1.jpg",
#   anti_spoofing = True
# )
# assert all(face_obj["is_real"] is True for face_obj in face_objs)

# # anti spoofing test in real time analysis
# DeepFace.stream(
#   db_path = "C:/User/Sefik/Desktop/database",
#   anti_spoofing = True
# )


#Real Time Analysis
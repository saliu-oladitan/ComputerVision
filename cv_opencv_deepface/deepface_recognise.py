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

dfs = DeepFace.find(
  img_path = "cv_opencv_deepface/cv_images/elon/elon_img2.jpg",
  db_path = "cv_opencv_deepface/cv_images",
  model_name = models[2],
  detector_backend = backends[0],
  align = alignment_modes[0],
  enforce_detection=False # Without this, it throws an error if a face isn't detected
)

print(" ")
print(" ")
print(dfs)


#anti_spoofing = True
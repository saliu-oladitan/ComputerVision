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


face_objs = DeepFace.extract_faces(
  img_path = "cv_opencv_deepface/cv_images/multifaces/multi4.jpg",
  detector_backend = backends[4],
  align = alignment_modes[0],
  anti_spoofing = True,
  enforce_detection=False
)

print(len(face_objs))
print(" ")
print(" ")
print(face_objs)


# Detect the faces, count them and tell which is real or fake
# Without this: anti_spoofing = True, it won't tell whether it's real or fake

# Once the face is coloured and looks like human, even if it's a cartoon it sees it as real except for
# extremely distorted or cartoonised images
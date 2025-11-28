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

embedding_objs = DeepFace.represent(
  img_path = "cv_opencv_deepface/cv_images/elon/elon_img1.jpg",
  detector_backend = backends[0],
  align = alignment_modes[0],
  enforce_detection=False # Without this, it throws an error if a face isn't detected
)

print(" ")
print(" ")
print(embedding_objs)


# Note: You need dlib and you need cmake for dlib
# Note: Restart VS code after installing cmake and other tools if they don't work after installation
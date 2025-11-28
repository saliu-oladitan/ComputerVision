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


demographies = DeepFace.analyze(
  img_path = "cv_opencv_deepface/cv_images/saliu/saliu_img1.jpg", 
  detector_backend = backends[3],
  align = alignment_modes[0],
  actions = ['age', 'gender', 'race', 'emotion'],
  enforce_detection=False # Without this, it throws an error if a face isn't detected
)

print(" ")
print(" ")
print(demographies)


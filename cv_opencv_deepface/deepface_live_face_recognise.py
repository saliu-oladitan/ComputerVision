import threading
import cv2 
from deepface import DeepFace

cap = cv2.VideoCapture (0) # Since only one camera is present

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

face_match = False

reference_img = cv2.imread("cv_opencv_deepface/cv_images/elon/elon_img1.jpg")

def check_face(frame):
  global face_match

  try:
    if DeepFace.verify (frame, reference_img.copy())['verified']:
      face_match=True
    else:
      face_match=False
  except ValueError: 
    face_match=False
 
while True:

 ret, frame = cap.read()

 if ret:
    if counter % 30 == 0:

      try:
       threading.Thread(target=check_face, args=(frame.copy(),)).start() 
      except ValueError:
       pass       # DeepFace usually throws a ValueError if it can't find a face
    counter += 1

    if face_match:
      cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3) # GBR and not RGB

    else:
     cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("video", frame)

 key = cv2.waitKey(1)
 if key == ord("q"):
    break
 
cv2.destroyAllWindows()



# Note: It needs good reference picture and good lightening during use



#==========================================

# import threading
# import cv2
# from deepface import DeepFace

# cap = cv2.VideoCapture(0) # Only one camera

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# counter = 0

# face_match = False

# reference_img = cv2.imread("cv_opencv_deepface/cv_images/saliu/saliu_img1.jpg")


# def check_face(frame):
#     global face_match
#     try:
#         if DeepFace.verify(frame, reference_img.copy())['verified']:
#             face_match = True
#         else:
#             face_match = False 
#     except ValueError:
#         face_match = False


# while True:
#     ret, frame = cap.read()

#     if ret:
#         if counter % 30 == 0:
#             try:
#                 threading.Thread(target=check_face, args=(frame.copy(),)).start() # We added the comma so that Python can consider it as a tuple
#             except ValueError:
#                 pass            # DeepFace usually throws a ValueError if it can't find a face
#         counter +=1
    
#         if face_match:
#             cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3) # GBR and not RGB
#         else:
#             cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
#         cv2.imshow("video", frame)
    
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
    

# cv2.destroyAllWindows()
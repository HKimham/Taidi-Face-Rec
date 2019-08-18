import cv2
import os
# from mtcnn.mtcnn import MTCNN
from os.path import join
import face_recognition


gray_path = './faceImageGray'

''' 对于face_recognition，速度较mtcnn慢一点，但是精确度很高 '''
for root, dirs, files in os.walk('./faceImages'):
    for dir in dirs:
        if not os.path.exists(join(gray_path,dir)):
            os.makedirs(join(gray_path,dir))
    for file in files:
        img = cv2.imread(join(root,file))
        faces = face_recognition.face_locations(img)
        if len(faces) == 0:
            continue
        top, right, bottom, left = faces[0] # 默认只有一张人脸
        face_box = cv2.cvtColor(img[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
        # face_box = cv2.resize(face_box,(64,64),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(join(gray_path,root.split('\\')[-1],file), face_box)
    print(root.split('\\')[-1] + ' is done!')


# ''' 对于MTCNN，速度较快，但是有部分人脸识别不出 '''
# detector = MTCNN()
# for root, dirs, files in os.walk('./faceImages'):
#     for dir in dirs:
#         if not os.path.exists(join(gray_path,dir)):
#             os.makedirs(join(gray_path,dir))
#     for file in files:
#         img = cv2.imread(join(root,file))
#         faces = detector.detect_faces(img)
#         if len(faces) == 0:
#             continue
#         x, y, width, height = faces[0]['box']
#         face_box = cv2.cvtColor(img[y : y + height, x : x + width], cv2.COLOR_RGB2GRAY)
#       # face_box = cv2.resize(face_box,(64,64),interpolation=cv2.INTER_CUBIC)
#         cv2.imwrite(join(gray_path,root.split('\\')[-1],file), face_box)
#     print(root.split('\\')[-1] + ' is done!')

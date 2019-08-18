import cv2
import os

""" 清空文件夹函数 """
def  del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i) 
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

images_path = './faceImages'
face_name = input('请输入人脸对象的姓名拼音(全小写): ')
file_path = os.path.join(images_path, face_name)
if os.path.isdir(file_path):
    del_file(file_path)
else:
    os.mkdir(file_path)

file_num = 600 # 采集600张图片

"""运行opencv,调用默认摄像头进行拍照"""
cap = cv2.VideoCapture(0)  # 调用默认摄像头
print("Tips: 采集满{}张将自动退出,按q可手动退出".format(file_num))
num = 0
while True:
    num += 1
    ret, frame = cap.read()
    if ret:  # 若不出错则显示图像
        cv2.imshow("face", frame)  # 弹窗口
        cv2.imwrite(os.path.join(file_path,"{}.jpg".format(num)), frame)
        if num % 50 == 0:
            print("{}已采集{}张人脸照片".format(face_name,num))
        if num >= file_num:
            print("采集完毕,自动退出!")
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'): 
            print("手动退出!")
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

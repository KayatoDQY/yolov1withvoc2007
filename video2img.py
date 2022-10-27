from fileinput import filename
import cv2

video=cv2.VideoCapture("dataset\\video\\data.mp4")

ret,frame=video.read()
i=0
while ret:
    filename="dataset\\image\\"+str(i)+".png"
    if i%10==0:
        cv2.imwrite(filename,frame)
    ret,frame=video.read()
    i=i+1
print("共",i,"张")
video.release()
from fileinput import close
import cv2
import datetime
capture = cv2.VideoCapture(0)
capture.set(3, 288)
capture.set(4, 352)

if capture.isOpened() == False:
    print("camera open failed")
    exit()

capNum = int (0)  

while True:
    ret, frame = capture.read() 

    if not ret:
        print("Can't read camera")
        break

    cv2.imshow("ex01", frame)

    if cv2.waitKey(1) == ord('c'):
        a = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        file = 'C:/Users/user/Desktop/2024.10.26/data/test/'+ '3 ('+ str(capNum) +').jpg'
        img_captured = cv2.imwrite(str(file),frame)
        capNum += 1
        print('captured')
            
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
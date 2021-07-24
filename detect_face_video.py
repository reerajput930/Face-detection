# we are doing face detection 
# we also copied the haar cascades face detector in xml file which is free for user use from github
# Face detection using Haar cascades is a machine learning based approach where a cascade function is trained with a set of input data.
#  OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc..
import cv2 as cv

# detect_face_video.py can detect more than one face
# try in high light room for better result

cap = cv.VideoCapture(0)


# loading the data of cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    isTrue, frame =cap.read()
    # cv.imshow("frame",frame) show normal frame


    # changing it to grayscale, reason: beacuse in face detection ,color in picture not needed
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # cv.imshow("gray",gray) show gray frame
 

    # detectin the face, detection happen in rectangular form
    # detectMultiScale(image, scale , noise detecter(senstive to edges lines))
    face_rect = face_cascade.detectMultiScale(gray,1.1,10)



    # marking the detected part
    for (x,y,w,h) in face_rect:
        # (image, starting width and height, end width and height,color,thikness)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,250,0),3)

    # try in high light room for better result
    cv.imshow("video_face_detect",frame)
    if(cv.waitKey(4)==ord('q')):
        break


# cv.waitKey(0)
cap.release()
cv.destroyAllWindows()

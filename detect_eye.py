# we are doing face detection 
# we also copied the haar cascades face detector in xml file which is free for user use from github
# Face detection using Haar cascades is a machine learning based approach where a cascade function is trained with a set of input data.
#  OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc..
import cv2 as cv

# detect_face.py can detect more than one face
# reading the images
img = cv.imread("./image_capture.jpg")


# changing it to grayscale, reason: beacuse in face detection ,color in picture not needed
gray1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# loading the data of cascade
face_cascade = cv.CascadeClassifier('haarcascade_eyedetect.xml')

# detectin the face, detection happen in rectangular form
# detectMultiScale(image, scale , noise detecter(senstive to edges lines))
face_rect = face_cascade.detectMultiScale(gray1,1.1,4)



# marking the detected part
for (x,y,w,h) in face_rect:
    # (image, starting width and height, end width and height,color,thikness)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,250,0),3)




cv.imshow("image1",img)


cv.waitKey(0)

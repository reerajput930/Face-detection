import cv2 as cv

# capture your face in it and try running detect_face.py to detect the face
# detect_face.py can detect more than one face
cap = cv.VideoCapture(0)

while True:
    isTrue, frame = cap.read()
    cv.imshow('video',frame)
    
    # waitkey related to frame speed and ord('q')means quit when press q keyward
    if cv.waitKey(4)==ord('q'):
        background = cv.imwrite("image_capture.jpg",frame)
        break

cap.release()
cv.destroyAllWindows()
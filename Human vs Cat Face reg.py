import cv2
import os
from PIL import Image
#cascade is what needed for face recognition
catcascade = cv2.CascadeClassifier('catface_detector.xml')
humancascade = cv2.CascadeClassifier('human_face.xml')

newsize = (600,600)
petcat = Image.open('Birdkiller.jpg')
petcat = petcat.resize(newsize)
petcat.save('Birdkillerresized.jpg')

humimg = Image.open('Charli at Party.jpg')
humimg = humimg.resize(newsize)
humimg.save('Charli at Partyresized.jpg')

#read by cv2
petcat1 = cv2.imread('Birdkillerresized.jpg')
humimg1 = cv2.imread('Charli at Partyresized.jpg')

cv2.imwrite('Birdkillerresized1.jpg', petcat1)
cv2.imwrite('Charli at Partyresized1.jpg', humimg1)

#detects the faces
catface = catcascade.detectMultiScale(petcat1,scaleFactor=1.3,minNeighbors=None, minSize=(75, 75))
humanface = humancascade.detectMultiScale(humimg1,scaleFactor=1.1,minNeighbors=None, minSize=(50,50))

#creates the recentangle 
for (x, y, w, h) in catface:
    cv2.rectangle(petcat1,(x,y),(x+w, y+h),(0, 255),2)

for (x, y, w, h) in humanface:
    cv2.rectangle(humimg1,(x, y),(x+w, y+h),(255, 0 ,0),4)

#shows the image
cv2.imshow ('face detected',humimg1)
cv2.imshow('cat faces', petcat1)
cv2.waitKey(0)
cv2.destroyAllWindows()

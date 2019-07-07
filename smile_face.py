
import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

img_counter = 0
while True:
    ret,img = cap.read()
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey)
    for face in  faces:
        x,y,w,h = face
        
        gray = grey[y:y + h, x:x + w]  

        smiles = smile_cascade.detectMultiScale(gray, 1.8, 20) 
  
        for (sx, sy, sw, sh) in smiles: 
            
            img_name = "Clear_face_{}.png".format(img_counter)
            cv2.imwrite(img_name, img)
            
            
            print("{} written!".format(img_name))
            
            img_counter += 1
    
    
    cv2.imshow('Face',img)
   
    cv2.waitKey(6)

cap.release()
cv2.destroyAllWindows()
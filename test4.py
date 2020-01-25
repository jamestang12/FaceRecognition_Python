from google.cloud import storage
from tempfile import NamedTemporaryFile
import cv2
import pickle
import time
import os
from twilio.rest import Client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/Owner/Downloads/MyProject.json"

count = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}
with open("labels.pickle",'rb') as f:   #f stand for file
   og_labels = pickle.load(f)
   labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
start = time.time()

while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_ , conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<= 85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y) , font, 1,color , stroke,cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, "unknown" , (x, y), font, 1, color, stroke, cv2.LINE_AA)
            if count == 0:
                client = Client("ACd6c7bfff08bea174c9cbe8c5a97d7838","548b3cb3e50cdda49c3d242f0c04ba4f")
                unknown_item = ("unknown_item.jpg")
                cv2.imwrite(unknown_item,roi_color)
                client2 = storage.Client()
                bucket = client2.get_bucket('jamestang')
                image = cv2.imread('unknown_item.jpg')
                with NamedTemporaryFile() as temp:
                    iName = "".join([str(temp.name), ".jpg"])
                    cv2.imwrite(iName, image)
                    blob = bucket.blob('unknown.jpg')
                    blob.upload_from_filename(iName, content_type='image/jpeg')
                    url = "https://storage.googleapis.com/jamestang/unknown.jpg"
                #client.messages.create(to="+16476068928" , from_="+18329003086", body="Unknown Visitor",media_url=media)
                #to = '+16476068928'
                #from1 = '+18329003086'
                to = 'whatsapp:+85263031883'
                from1 = 'whatsapp:+14155238886'
                client.messages.create(body="Unknown Visitor", from_=from1,to=to,media_url=url)
                print("Unknown")
                count = count + 2
            elif count != 0:
                stop = time.time()
                if(stop - start > 6):
                    client = Client("ACd6c7bfff08bea174c9cbe8c5a97d7838", "548b3cb3e50cdda49c3d242f0c04ba4f")
                    unknown_item = ("unknown_item.jpg")
                    image = cv2.imread('unknown_item.jpg')
                    cv2.imwrite(unknown_item, roi_color)
                    with NamedTemporaryFile() as temp2:
                        iName2 = "".join([str(temp.name), ".jpg"])
                        cv2.imwrite(iName2, image)
                        blob2 = bucket.blob('unknown.jpg')
                        blob2.upload_from_filename(iName, content_type='image/jpeg')
                        #url = "https://cdn.images.express.co.uk/img/dynamic/171/590x/John-Cena-1220628.jpg?r=1577148479218"
                        url = "https://storage.googleapis.com/jamestang/unknown.jpg"
                    to = 'whatsapp:+85263031883'
                    from1 = 'whatsapp:+14155238886'
                    client.messages.create(body="Unknown Visitor", from_=from1, to=to, media_url=url)
                    print(url)
                    print("Unknown")
                    start = time.time()


        img_item = 'my-image.jpg'
        cv2.imwrite(img_item,roi_color)

        color = (255,0,0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y) , (end_cord_x,end_cord_y), color,stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
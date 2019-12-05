import cv2
cascade_src = 'monitor.xml'
CASCADE_ITEM = 'monitor'
cap = cv2.VideoCapture('monitor.mp4')

car_cascade = cv2.CascadeClassifier(cascade_src)


while True:
    ret, img = cap.read()
   
    if (type(img) == type(None)):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, -50, minSize=(250, 250))


    for (i, (x, y, w, h)) in enumerate(cars):
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img, CASCADE_ITEM + " #{}".format(i + 1), (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imshow(CASCADE_ITEM, img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()

import cv2
import pytesseract

import datetime

now = datetime.datetime.now()


frameWidth = 640
frameHeight = 480

plateCascade = cv2.CascadeClassifier(r"D:\\monish\\haarcascade_russian_plate_number.xml")
minArea = 500

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.putText(frame, "SOLUTION FRAME", (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

    contours, h = cv2.findContours(thresh, 1, 2)
    largest_rectangle = [0, 0]
    for cnt in contours:
        lenght = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, lenght, True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > largest_rectangle[0]:
                largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
    x, y, w, h = cv2.boundingRect(largest_rectangle[1])

    image = frame[y:y + h, x:x + w]
    cv2.drawContours(frame, [largest_rectangle[1]], 0, (0, 255, 0), 8)
    cropped = frame[y:y + h, x:x + w]
    cv2.putText(frame, "LICENSE PLATE", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 0, 255))
    cv2.imshow('Locating License Plates', frame)
    cv2.drawContours(frame, [largest_rectangle[1]], 0, (255, 255, 255), 18)

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow('Result', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    print("numberplate:")
    print(data, now)

    key = cv2.waitKey(500)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

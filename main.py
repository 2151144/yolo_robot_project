from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ لم يتم فتح الكاميرا")
    exit()

print("✅ تم تشغيل الكاميرا بنجاح")
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ لم يتم التقاط الصورة")
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

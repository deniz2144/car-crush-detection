import cv2
import numpy as np


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()  
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] .


with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()] 


def detect_objects(frame):
    height, width, channels = frame.shape  
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  
    net.setInput(blob)  
    outs = net.forward(output_layers) 
    
    class_ids = []  
    confidences = []  
    boxes = []  
    centers = []  
    
    for out in outs:  
        for detection in out: 
            scores = detection[5:] 
            class_id = np.argmax(scores)  
            confidence = scores[class_id]  
            if confidence > 0.5 and classes[class_id] == "car": 
                center_x = int(detection[0] * width) 
                center_y = int(detection[1] * height) 
                w = int(detection[2] * width)
                h = int(detection[3] * height) 
                x = int(center_x - w / 2) 
                y = int(center_y - h / 2)  
                boxes.append([x, y, w, h])  
                centers.append((center_x, center_y))  
                confidences.append(float(confidence))  
                class_ids.append(class_id)  
    
    return boxes, centers  

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)  # İki nokta arasındaki Öklid mesafesini hesapla.

# Çarpışma tespiti fonksiyonu (sadece mesafeyi kullanarak)
def detect_collision(centers, distance_threshold=50):  
    for i in range(len(centers)): 
        for j in range(i + 1, len(centers)): 
            distance = calculate_distance(centers[i], centers[j])  .
            
           
            if distance < distance_threshold:  
                print("Collision detected!")  .
                return True  #
    return False  

cap = cv2.VideoCapture("crush.mp4")

collision_display_time = 0 
collision_duration = 30 

while cap.isOpened(): 
    ret, frame = cap.read()  
    if not ret:  
        break  
    
    detected_boxes, car_centers = detect_objects(frame)  
    
    if detect_collision(car_centers):  .
        collision_display_time = collision_duration
    
    if collision_display_time > 0: 
        cv2.putText(frame, "Kaza tespit edildi!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  
        collision_display_time -= 1  .
    
    for box, center in zip(detected_boxes, car_centers):  
        x, y, w, h = box  
        center_x, center_y = center  
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  .
        cv2.putText(frame, f'({center_x},{center_y})', (center_x - 20, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  
    
    cv2.imshow("Frame", frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

cap.release()  
cv2.destroyAllWindows()  

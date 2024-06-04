import cv2
import numpy as np

# YOLOv3 modelini ve konfigürasyon dosyasını yükle.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()  # Modelin tüm katman isimlerini al.
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Çıkış katmanlarını al.

# COCO veri seti sınıf isimlerini yükle.
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]  # Dosyadaki her satırı sınıf olarak listeye ekle.

# Nesne tespit fonksiyonu
def detect_objects(frame):
    height, width, channels = frame.shape  # Frame'in boyutlarını al.
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Resmi YOLOv3 modeline uygun şekilde blob'a dönüştür.
    net.setInput(blob)  # Blob'u modelin girişi olarak ayarla.
    outs = net.forward(output_layers)  # Modeli ileri geçirerek çıktı katmanlarını al.
    
    class_ids = []  # Tespit edilen sınıf ID'lerini sakla.
    confidences = []  # Tespitlerin güven skorlarını sakla.
    boxes = []  # Tespit edilen nesnelerin kutu koordinatlarını sakla.
    centers = []  # Tespit edilen nesnelerin merkez noktalarını sakla.
    
    for out in outs:  # Her bir çıktı katmanı için.
        for detection in out:  # Her bir tespit için.
            scores = detection[5:]  # Tespit edilen nesnenin sınıf skorlarını al.
            class_id = np.argmax(scores)  # En yüksek skoru alan sınıf ID'sini bul.
            confidence = scores[class_id]  # En yüksek skorun güven değerini al.
            if confidence > 0.5 and classes[class_id] == "car":  # Güven değeri 0.5'ten büyükse ve sınıf araba ise.
                center_x = int(detection[0] * width)  # Tespitin merkez x koordinatını bul.
                center_y = int(detection[1] * height)  # Tespitin merkez y koordinatını bul.
                w = int(detection[2] * width)  # Tespitin genişliğini bul.
                h = int(detection[3] * height)  # Tespitin yüksekliğini bul.
                x = int(center_x - w / 2)  # Kutunun sol üst köşesinin x koordinatını bul.
                y = int(center_y - h / 2)  # Kutunun sol üst köşesinin y koordinatını bul.
                boxes.append([x, y, w, h])  # Kutuyu listeye ekle.
                centers.append((center_x, center_y))  # Merkez noktasını listeye ekle.
                confidences.append(float(confidence))  # Güven değerini listeye ekle.
                class_ids.append(class_id)  # Sınıf ID'sini listeye ekle.
    
    return boxes, centers  # Tespit edilen kutuları ve merkezleri döndür.

# Mesafe hesaplama fonksiyonu
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)  # İki nokta arasındaki Öklid mesafesini hesapla.

# Çarpışma tespiti fonksiyonu (sadece mesafeyi kullanarak)
def detect_collision(centers, distance_threshold=50):  # Mesafe eşiği piksel cinsindendir.
    for i in range(len(centers)):  # Her merkez noktası için.
        for j in range(i + 1, len(centers)):  # Diğer merkez noktalarıyla karşılaştır.
            distance = calculate_distance(centers[i], centers[j])  # İki merkez arasındaki mesafeyi hesapla.
            
           
            if distance < distance_threshold:  # Mesafe eşiğin altındaysa.
                print("Collision detected!")  # Çarpışma tespit edildiğini yazdır.
                return True  # Çarpışma tespit edildiğini döndür.
    return False  # Çarpışma tespit edilmediğini döndür.

# Video dosyasını aç
cap = cv2.VideoCapture("crush.mp4")

collision_display_time = 0  # Çarpışma mesajının gösterildiği süreyi başlat.
collision_duration = 30  # Çarpışma mesajının kaç kare boyunca gösterileceğini ayarla.

while cap.isOpened():  # Video dosyası açık olduğu sürece.
    ret, frame = cap.read()  # Videodan bir kare oku.
    if not ret:  # Eğer kare okunamadıysa.
        break  # Döngüden çık.
    
    detected_boxes, car_centers = detect_objects(frame)  # Karedeki araçları tespit et.
    
    if detect_collision(car_centers):  # Eğer çarpışma tespit edildiyse.
        collision_display_time = collision_duration  # Çarpışma mesajını gösterme süresini başlat.
    
    if collision_display_time > 0:  # Eğer çarpışma mesajı gösteriliyorsa.
        cv2.putText(frame, "Kaza tespit edildi!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)  # Çarpışma mesajını ekrana yazdır.
        collision_display_time -= 1  # Çarpışma mesajı süresini bir azalt.
    
    for box, center in zip(detected_boxes, car_centers):  # Tespit edilen her kutu ve merkez için.
        x, y, w, h = box  # Kutunun koordinatlarını al.
        center_x, center_y = center  # Merkez noktasını al.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Kutuyu çiz.
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Merkez noktasını işaretle.
        cv2.putText(frame, f'({center_x},{center_y})', (center_x - 20, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Merkez koordinatını yazdır.
    
    cv2.imshow("Frame", frame)  # Kareyi ekranda göster.
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q tuşuna basılırsa.
        break  # Döngüden çık.

cap.release()  # Video yakalamayı serbest bırak.
cv2.destroyAllWindows()  # Pencereleri kapat.

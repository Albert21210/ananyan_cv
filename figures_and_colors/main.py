import cv2
import numpy as np

image = cv2.imread("balls_and_rects.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

colors = {
    "Красный": ([0, 50, 50], [10, 255, 255]),
    "Оранжевый": ([11, 50, 50], [25, 255, 255]),
    "Жёлтый": ([26, 50, 50], [35, 255, 255]),
    "Зелёный": ([36, 50, 50], [85, 255, 255]),
    "Синий": ([86, 50, 50], [130, 255, 255]),
    "Фиолетовый": ([131, 50, 50], [160, 255, 255]),
    "Розовый": ([161, 50, 50], [180, 255, 255])
}

total_figures = 0
results = {}

for color_name, (lower, upper) in colors.items():
    lower = np.array(lower)
    upper = np.array(upper)
    
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = 0
    rects = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: 
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: 
            continue
        
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        if circularity > 0.85: 
            balls += 1
        else:
            rects += 1
            
        total_figures += 1
    
    if balls > 0 or rects > 0:
        results[color_name] = {"Круги": balls, "Прямоугольники": rects}

print(f"Общее количество фигур: {total_figures}")
print("-" * 30)
for color, counts in results.items():
    print(f"Оттенок: {color}")
    print(f"   Кругов: {counts['Круги']}")
    print(f"   Прямоугольников: {counts['Прямоугольники']}")

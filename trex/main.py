import mss
import numpy as np
import pyautogui
import time
import cv2

# Координаты области поиска
GAME_REGION = {'top': 320, 'left': 675, 'width': 700, 'height': 100}

# Масштаб окна
SCALE = 4

def dino():
    print("Бот запущен. Переключитесь на окно с игрой!")
    time.sleep(3)

    # Засекаем время
    start_time = time.perf_counter()
    last_update_time = start_time
    step = 0
    
    with mss.mss() as sct:
        # Определяем фоновый цвет
        sample = np.array(sct.grab(GAME_REGION))
        background_color = sample[0, 0, 0] 
        
        while True:
            current_time = time.perf_counter()
            total_time = current_time - start_time

            # Смещение области со временем
            if current_time - last_update_time >= 8:
                if total_time <= 9: step = 10
                elif total_time <= 17: step = 11
                elif total_time <= 25: step = 11
                elif total_time <= 33: step = 13
                elif total_time <= 42: step = 13
                elif total_time <= 50: step = 15
                elif total_time <= 58: step = 18
                elif total_time <= 67: step = 18
                elif total_time <= 75: step = 17
                elif total_time <= 83: step = 15
                elif total_time <= 92: step = 14
                elif total_time <= 100: step = 12
                elif total_time <= 108: step = 13
                elif total_time <= 117: step = 33
                else: step = 0
                
                last_update_time = current_time 
                GAME_REGION['left'] += step
                print(f"ОБНОВЛЕНИЕ: Прошло: {int(total_time)} сек. Добавили: {step}. Теперь Left: {GAME_REGION['left']}")

            # Захват кадра
            img = np.array(sct.grab(GAME_REGION))
            debug_img = img.copy()
            
            gray = img[:, :, 0]
            
            # Зона обнаружения препятсвий
            roi_birds = gray[31:33, 100:250]
            roi_obstacles = gray[65:75, 100:250]
            
            # Находим все пиксели, которые НЕ являются фоном
            birds_mask = roi_birds != background_color
            obstacle_mask = roi_obstacles != background_color

            birds_count = np.sum(birds_mask)
            obstacle_count = np.sum(obstacle_mask)

            # Рисуем зеленый прямоугольник для зоны птиц
            cv2.rectangle(debug_img, (100, 31), (250, 33), (0, 255, 0), 1)
            # Рисуем красный прямоугольник для зоны кактусов
            cv2.rectangle(debug_img, (100, 65), (250, 75), (0, 0, 255), 1)

            # Подсвечиваем найденные пиксели внутри зон
            debug_img[31:33, 100:250][birds_mask] = [0, 255, 0, 255]
            debug_img[65:75, 100:250][obstacle_mask] = [0, 0, 255, 255]

            # Формула размера окна
            new_width = int(debug_img.shape[1] * SCALE)
            new_height = int(debug_img.shape[0] * SCALE)
            
            # Масштабируем окно
            resized_img = cv2.resize(debug_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Dino", resized_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Проверка птиц на средней высоте
            if birds_count > 0:
                print(f"Птица!")
                pyautogui.keyDown('down')
                time.sleep(0.5)
                pyautogui.keyUp('down')
            
            # Если в зоне есть хоть какое-то препятствие
            if obstacle_count > 0:
                obstacle_pixels_x = np.where(obstacle_mask)[1]
                min_x = np.min(obstacle_pixels_x)

                if min_x <= 25:
                    # Прыжок для маленького одинарного кактуса
                    if obstacle_count <= 210:
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('up')
                        time.sleep(0.0015)
                        pyautogui.keyDown("down")
                        pyautogui.keyUp("down")
                        print(f"Маленький одинарный кактус!")

                    # Прыжок для большого одинарного кактуса и птицы
                    elif obstacle_count > 210 and obstacle_count <= 380:
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('up')
                        time.sleep(0.0195) 
                        pyautogui.keyDown("down")
                        pyautogui.keyUp("down")
                        print(f"Большой одинарный кактус или птица!")

                    # Прыжок для маленького и большого двойного кактуса
                    elif obstacle_count > 380 and obstacle_count <= 480:
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('up')
                        time.sleep(0.0785) 
                        pyautogui.keyDown("down")
                        pyautogui.keyUp("down")
                        print(f"Маленький или большой двойной кактус!")

                    # Прыжок для маленького тройного кактуса
                    elif obstacle_count > 480 and obstacle_count <= 670:
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('up')
                        time.sleep(0.0805) 
                        pyautogui.keyDown("down")
                        pyautogui.keyUp("down")
                        print(f"Маленький тройной кактус!")

                    # Прыжок для большого четверного кактуса
                    elif obstacle_count > 670:
                        pyautogui.keyDown('up')
                        pyautogui.keyUp('up')
                        time.sleep(0.14) 
                        pyautogui.keyDown("down")
                        pyautogui.keyUp("down")
                        print(f"Большого четверной кактус!")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    dino()

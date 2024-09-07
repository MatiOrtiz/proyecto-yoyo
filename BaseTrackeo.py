import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###################################################
#      PARTE DE TRACKEO DE POR COLOR DE OPENCV    #
###################################################

# Define los límites del color en el espacio HSV
greenLower = (45, 50, 100)
greenUpper = (85, 255, 255)

# Carga el video
cap = cv2.VideoCapture('verde.mp4')

# Variables para guardar las posiciones y tiempos
positions = []
fps = cap.get(cv2.CAP_PROP_FPS)  # Obtiene los FPS del video
frame_duration = 1.0 / fps  # Duración de cada frame en segundos
frame_count = 0

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convierte el frame al espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crea una máscara para el color verde
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    
    # Realiza una serie de dilataciones y erosiones para eliminar cualquier pequeño punto en la máscara
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Encuentra los contornos en la máscara
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encuentra el contorno más grande en la máscara
        c = max(contours, key=cv2.contourArea)
        
        # Encuentra el centro del contorno
        M = cv2.moments(c)
        if M["m00"] > 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            center = (center_x, center_y)
            
            # Calcula el tiempo actual basado en el número de frames procesados
            current_time = frame_count * frame_duration
            positions.append((current_time, center_x, center_y))
            
            # Dibuja el contorno y el centro en el frame original
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    # Muestra el frame con el tracking
    cv2.imshow('Object Tracking', frame)
    
    # Rompe el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Incrementa el contador de frames
    frame_count += 1

# Libera la captura y cierra las ventanas
cap.release()
cv2.destroyAllWindows()

##########################################################################################
#                                CREACION DE DATAFRAME                                   #
##########################################################################################
# Crea un DataFrame con las posiciones en funcion del tiempo que capturo el codigo de opencv
df = pd.DataFrame(positions, columns=['Time (sec)', 'X', 'Y'])

# Calcula las diferencias entre posiciones y tiempos, lo agrega al dataframe
#diff: es una funcion que calcula la diferencia entre los elementos consecutivos, por ejemplo de tiempo en este caso.
#fillna(0): remplaza los elementos vacios por un 0 para que no tire error.
df['Delta_Time'] = df['Time (sec)'].diff().fillna(0) 
df['Delta_X'] = df['X'].diff().fillna(0)
df['Delta_Y'] = df['Y'].diff().fillna(0)


# Calcula la velocidad en X y Y a partir de las diferencias en posicion y el cambio en el tiempo, lo agrega al dataframe
df['Speed_X'] = df['Delta_X'] / df['Delta_Time']
df['Speed_Y'] = df['Delta_Y'] / df['Delta_Time']

# Reemplaza inf y -inf por 0 en las velocidades en x e y (por si se va al infinito por dividir por cero por ejemplo)
df['Speed_X'] = df['Speed_X'].replace([np.inf, -np.inf], 0)
df['Speed_Y'] = df['Speed_Y'].replace([np.inf, -np.inf], 0)

# Calcula la velocidad total, lo agrega al dataframe
df['Speed'] = np.sqrt(df['Speed_X']**2 + df['Speed_Y']**2)

# Reemplaza inf y -inf por 0 en la velocidad total
df['Speed'] = df['Speed'].replace([np.inf, -np.inf], 0)

#############################################################################
# Guarda el DataFrame actualizado en un archivo CSV (es para chequear despues si todo esta en orden nomas)
df.to_csv('positions_with_speed.csv', index=False, float_format='%.6f')

print("Datos de posiciones y velocidades guardados en 'positions_with_speed.csv'.")

##############################################################################
#                         PLOTEO                                             #
##############################################################################
# Grafica la trayectoria y la velocidad (modulo)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Grafica la trayectoria
ax1.plot(df['X'], df['Y'], marker='o', linestyle='-', color='b')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Trayectoria del Objeto')

# Grafica la velocidad
ax2.plot(df['Time (sec)'], df['Speed'], marker='o', linestyle='-', color='r', label='Velocidad Total')
ax2.set_xlabel('Tiempo (segundos)')
ax2.set_ylabel('Velocidad (pixeles/segundo)')
ax2.set_title('Velocidad del Objeto')
ax2.legend()

# Ajustar el diseño
plt.tight_layout()

# Muestra los graficos
plt.show()

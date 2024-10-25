import numpy as np
import argparse
import cv2 as cv

def detectLaneLines(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Ajustar os intervalos de cor para uma detecção mais precisa
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv.inRange(hsv, lower_white, upper_white)
    
    mask = cv.bitwise_or(mask_yellow, mask_white)
    
    # Aplicar morfologia para limpar pequenas regiões indesejadas
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    result = cv.bitwise_and(image, image, mask=mask)

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv.Canny(blurred, 50, 150)
    
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    # Desenhar linhas detectadas
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return image

def detectObjectsInImage(imagePath):
    image = cv.imread(imagePath)
    image_with_lanes = detectLaneLines(image)
    cv.imshow("Imagem com Faixas de Trânsito Detectadas", image_with_lanes)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detectObjectsInVideo(videoPath=None, camera_index=None, resize_factor=1.0):
    if videoPath:
        cap = cv.VideoCapture(videoPath)
    elif camera_index is not None:
        cap = cv.VideoCapture(camera_index)
    else:
        print("Por favor, forneça o caminho para um vídeo usando -v ou o índice da câmera usando -c.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if resize_factor != 1.0:
            frame = cv.resize(frame, None, fx=resize_factor, fy=resize_factor)

        image_with_lanes = detectLaneLines(frame)
        cv.imshow("Vídeo com Faixas de Trânsito Detectadas", image_with_lanes)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="Caminho para a imagem")
    ap.add_argument("-v", "--video", help="Caminho para o vídeo")
    ap.add_argument("-c", "--camera", type=int, default=0, help="Índice da câmera (0 para webcam padrão)")
    ap.add_argument("-r", "--resize", type=float, default=1.0, help="Fator de redimensionamento do frame (0-1 para reduzir)")

    args = vars(ap.parse_args())

    if args["image"]:
        detectObjectsInImage(args["image"])
    elif args["video"]:
        detectObjectsInVideo(videoPath=args["video"], resize_factor=args["resize"])
    elif args["camera"] is not None:
        detectObjectsInVideo(camera_index=args["camera"], resize_factor=args["resize"])
    else:
        print("Por favor, forneça o caminho para uma imagem usando -i, para um vídeo usando -v, ou o índice da câmera usando -c.")

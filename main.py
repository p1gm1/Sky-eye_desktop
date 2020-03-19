import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops,label
import glob
import tkinter.filedialog as fdlg
from tkinter import Tk

#Cargar imagenes por lotes
def batch_img_load(dir):
    cv_img = []
    for img in glob.glob(dir):
        n = cv2.imread(img)
        cv_img.append(n)
    return cv_img


root = Tk().withdraw()
dir_name = fdlg.askdirectory()
imag = batch_img_load(dir_name +'/*.jpg')

#Cargar una única imagen
# imag= cv2.imread('IMG_181101_155057_0000_RGB.JPG')

#Guardar en archivo de texto
# arq = fdlg.asksaveasfile(mode='w',defaultextension=".txt",filetypes = (("Text file", "*.txt"),("CSV File","*.csv")))
arq = open("Individuos_Ordenados2.txt",'a')
arq.write('\n\nCantidad de individuos en imagen: '+ dir_name +'\n')

#Declarando variable de suma
sum_p = 0

for m in range(len(imag)):
    #Separación en dos canales RGB
    # (B, G, R) = cv2.split(imag[m])

    #Transformando la separación a HSV
    hsv = cv2.cvtColor(imag[m], cv2.COLOR_BGR2HSV)
    (H, S, V) = cv2.split(hsv)

    #Umbralizaicón
    _, imgbin = cv2.threshold(H, 22, 255, cv2.THRESH_BINARY)

    #Dibujar histograma
    # hist = plt.hist(H.ravel(),256,[0,255])
    # plt.show()

    #Dilatación y erosión
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], 
        [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], np.uint8)
    kernel3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    kernel4 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    imgbin = cv2.erode(imgbin, kernel4, iterations=2)
    imgbin = cv2.dilate(imgbin, kernel4, iterations=2)

    #Separación de fondo y objeto
    for i in range(imag[m].shape[0]):
        for j in range(imag[m].shape[1]):
            logic = imgbin.item(i, j)
            if logic == 0:
                imag[m].itemset((i, j, 2), 0)
                imag[m].itemset((i, j, 1), 0)
                imag[m].itemset((i, j, 0), 0)

    #Redimensionamiento de imagen
    img_resize = cv2.resize(imag[m], 
        (int(imag[m].shape[1] * 0.3), int(imag[m].shape[0] * 0.3)))

    #Guardar imagem
    cv2.imwrite('mod.JPG', imgbin)

    #Cantidad de indivíduos
    label_image, num = label(imgbin, return_num=True)
    regions = regionprops(label_image)
    areas = [r.area for r in regions]
    areas.sort()
    print(areas)
    # t = 0
    # while t < len(areas):
    #     if areas[t] > 7000:
    #         areas.pop(t)
    #     else:
    #         t += 1

    quant = len(areas)
    print('La cantidad de indivíduos encontrados en la imagen es: ', quant)

    #Sumar total de indíviduos
    sum_p += quant

    #Escribir en archivo
    arq.write('\nImagen (' + str(m+1) + ') - Individuos: ' + str(quant))

    #Mostrar imagem
    cv2.imshow('Imagen', img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Escribir el numero final de individuos
arq.write('\nTotal de individuos = ' + str(sum_p))
print('Total de indíviduos = ', sum_p)

#Archivo de cierre
arq.close()
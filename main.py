import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops,label
import glob
import tkinter.filedialog as fdlg
from tkinter import Tk

#Carregar imagem em lotes
def carregar_imagens_em_lote(diretorio):
    cv_img = []
    for img in glob.glob(diretorio):
        n = cv2.imread(img)
        cv_img.append(n)
    return cv_img

root = Tk().withdraw()
pasta = fdlg.askdirectory()
imagem = carregar_imagens_em_lote(pasta+'/*.jpg')

#Carregar imagens única
# imagem= cv2.imread('IMG_181101_155057_0000_RGB.JPG')

#Salvar em arquivo de texto
# arq = fdlg.asksaveasfile(mode='w',defaultextension=".txt",filetypes = (("Text file", "*.txt"),("CSV File","*.csv")))
arq = open("Individuos_Zortea2.txt",'a')
arq.write('\n\nQuantidade de individuos da pasta: '+ pasta+'\n')

#Declaração variável de soma
soma=0

for m in range(len(imagem)):
    #Separação dos canais RGB
    # (B, G, R) = cv2.split(imagem[m])

    #Transformação e separação do HSV
    hsv = cv2.cvtColor(imagem[m], cv2.COLOR_BGR2HSV)
    (H, S, V) = cv2.split(hsv)

    #Limiarização
    _, imgbin = cv2.threshold(H, 22, 255, cv2.THRESH_BINARY)

    #Plotar histograma
    # hist = plt.hist(H.ravel(),256,[0,255])
    # plt.show()

    #Dilatação e erosão
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], np.uint8)
    kernel3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    kernel4 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    imgbin = cv2.erode(imgbin, kernel4, iterations=2)
    imgbin = cv2.dilate(imgbin, kernel4, iterations=2)

    #Separação fundo/objeto
    for i in range(imagem[m].shape[0]):
        for j in range(imagem[m].shape[1]):
            logica = imgbin.item(i, j)
            if logica == 0:
                imagem[m].itemset((i, j, 2), 0)
                imagem[m].itemset((i, j, 1), 0)
                imagem[m].itemset((i, j, 0), 0)

    #Redimensionamento da imagem
    redmensionada = cv2.resize(imagem[m], (int(imagem[m].shape[1] * 0.3), int(imagem[m].shape[0] * 0.3)))

    #Salvar imagem
    cv2.imwrite('modificada.JPG', imgbin)

    #Quantidade de indivíduos
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

    quantidade = len(areas)
    print('A quantidade de indivíduos encontrados na imagem é: ', quantidade)

    #Somar total de indíviduos
    soma += quantidade

    #Escrever na arquivo
    arq.write('\nImagem (' + str(m+1) + ') - Individuos: ' + str(quantidade))

    #Mostra imagem
    cv2.imshow('Imagem', redmensionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Escrever quantidade final de indivíduos
arq.write('\nTotal de individuos = ' + str(soma))
print('Total de indíviduos = ', soma)

#Fechando arquivo
arq.close()
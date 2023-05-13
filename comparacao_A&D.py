import cv2
import face_recognition as fr

#Carregando e colorindo as imagens
imgAndre = fr.load_image_file('andre.png')
imgAndre = cv2.cvtColor(imgAndre, cv2.COLOR_BGR2RGB)
imgDaniel = fr.load_image_file('daniel.png')
imgDaniel = cv2.cvtColor(imgDaniel, cv2.COLOR_BGR2RGB)

#Localizando o rosto na imagem
faceLoc1 = fr.face_locations(imgAndre)[0]
cv2.rectangle(imgAndre, (faceLoc1[3], faceLoc1[0]), (faceLoc1[1], faceLoc1[2]), (7, 700, 2), 2) #o primeiro argumento é Y e o segundo o X
print(faceLoc1)

#Extraindo as 128 medidas do rosto na imagem
encodeAndre = fr.face_encodings(imgAndre)[0]
encodeDaniel = fr.face_encodings(imgDaniel)[0]

#Realizando a comparação dos dados da imagem para determinar se é a mesma pessoa
comparacao = fr.compare_faces([encodeAndre], encodeDaniel)

#Printando as imagens
print(comparacao)
cv2.imshow('Prof. Andre', imgAndre)
cv2.imshow('Prof. Daniel', imgDaniel)
cv2.waitKey(0)


#Importando os módulos necessários
import cv2
import face_recognition as fr

#Carregando as imagens
imgAndre1 = fr.load_image_file('andre.png')
imgAndre1 = cv2.cvtColor(imgAndre1, cv2.COLOR_BGR2RGB)
imgAndre2 = fr.load_image_file('andre2.png')
imgAndre2 = cv2.cvtColor(imgAndre2, cv2.COLOR_BGR2RGB)

#Localizando o rosto na imagem
faceLoc = fr.face_locations(imgAndre1)[0]
cv2.rectangle(imgAndre1, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (7, 700, 2), 2)
print(faceLoc)

#Extraindo as 128 medidas do rosto na imagem
encodeAndre1 = fr.face_encodings(imgAndre1)[0]
encodeAndre2 = fr.face_encodings(imgAndre2)[0]

#Realizando a comparação dos dados da imagem para determinar se é a mesma pessoa
comparacao = fr.compare_faces([encodeAndre1], encodeAndre2)

#Printando as imagens
print(comparacao)
cv2.imshow("Prof. Andre base", imgAndre1)
cv2.imshow("Teste Prof. Andre", imgAndre2)
cv2.waitKey(0)

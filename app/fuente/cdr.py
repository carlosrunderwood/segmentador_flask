import cv2
import app.fuente.copaSeg as cs
import app.fuente.bib as bib
import os

def caracteristicas(imagen, image_name):
    image_name = image_name.split(sep='.')
    if(not(os.path.isdir("app/static/imagenes/"+image_name[0]))):
        os.mkdir("app/static/imagenes/"+image_name[0])
    centroC, dhC, dvC, daC, mascaraC, corte = cs.copaSeg(imagen,image_name[0],"app/static/imagenes/"+image_name[0])
    suma = bib.suma(corte,mascaraC)
    cv2.imwrite("app/static/imagenes/"+image_name[0]+"/" +image_name[0]+ "_final.jpg",suma)
    #cv2.imwrite("" + dst + "/" + src + "_original.jpg",img)
    return 

#name = "N-83-L"
#caracteristicas(name)

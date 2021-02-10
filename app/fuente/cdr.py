import cv2
import app.fuente.copaSeg as cs
import app.fuente.discSeg as ds
import app.fuente.Glaucoma as pre
import app.fuente.bib as bib
import os


def caracteristicas(imagen,imagen_name,maskC):
    
    image_name = imagen_name.split(sep='.')
    if(not(os.path.isdir("app/static/imagenes/"+image_name[0]))):
        os.mkdir("app/static/imagenes/"+image_name[0])
    
    corte, repaint, veins, maskC = pre.preprocesamiento(imagen,image_name[0],maskC,"app/static/imagenes/"+image_name[0])
    
    centroD, dhD, dvD,maskD = ds.discSeg(repaint,veins,image_name[0],"app/static/imagenes/"+image_name[0])
    
    centroC, dhC, dvC, mascaraC, corte, dice, salto = cs.copaSeg(corte,image_name[0],maskC,maskD,"app/static/imagenes/"+image_name[0])
    

    suma = bib.suma(corte,mascaraC) 
    cv2.imwrite("app/static/imagenes/"+image_name[0] + "/"+image_name[0]+ "_final.jpg",suma)
    cv2.imwrite("app/static/imagenes/"+image_name[0] + "/"+image_name[0]+ "_Experto.jpg",maskC)
    
    return dice

def caracteristicasSD(imagen,imagen_name):
    
    image_name = imagen_name.split(sep='.')
    if(not(os.path.isdir("app/static/imagenes/"+image_name[0]))):
        os.mkdir("app/static/imagenes/"+image_name[0])
    
    corte, repaint, veins = pre.preprocesamientoSD(imagen,image_name[0],"app/static/imagenes/"+image_name[0])
    
    centroD, dhD, dvD,maskD = ds.discSeg(repaint,veins,image_name[0],"app/static/imagenes/"+image_name[0])
    
    centroC, dhC, dvC, mascaraC, corte, salto = cs.copaSegSD(corte,image_name[0],maskD,"app/static/imagenes/"+image_name[0])
    
    
    suma = bib.suma(corte,mascaraC) 
    cv2.imwrite("app/static/imagenes/"+image_name[0] + "/"+image_name[0]+ "_final.jpg",suma)
    
    return 
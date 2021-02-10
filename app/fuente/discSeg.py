import cv2
import app.fuente.preprocesamiento as pre
import app.fuente.bib as bib


def discSeg(src,veins, name, dst):
    stage = "OD-seg"
    img = src
    shape = img.shape
        
    w,h = shape[:2]
    
    if( (w < h/2) or (h < w/2) ):
        print("omitiendo la segmentacion de disco optico: " + name)
        return None,None,None,None,None
    
    _, _,R = pre.BGR(img)
    
    homo = pre.Homomorfico(R,100)
    cv2.imwrite(dst + "/"+name + "_" + stage + "_1_Homomorfico.jpg",homo)
    
    rob = pre.robustecer(homo,7,3)
    cv2.imwrite(dst + "/"+name + "_" + stage + "_2_Robusto.jpg",rob)
    
    otsu = bib.umbralizar(rob, 93)
    cv2.imwrite(dst + "/"+name + "_" + stage + "_3_GlobalT.jpg",otsu)
    
    apertura = bib.apertura(otsu,27)
    # cierre = bib.cierre(Apertura,15)
    
    cv2.imwrite(dst + "/"+name + "_" + stage + "_4_Apertura.jpg",apertura)
    
    
    candidato = bib.candidateElection(apertura, veins)
    
    candidato = pre.threshold(candidato,200)
    
    candidato = bib.cierre(candidato,400)
    
    contornos,indice = bib.bordesCanny(candidato)
    
    cont = bib.drawContours(contornos, shape, indice)
    
    _, dv,dh, center = bib.boundingBox(contornos, shape, indice)
    # print(indice, len(contornos))
    
    candidato = cv2.merge([candidato, candidato, candidato])
    
    final = bib.suma(img,cont)
    # cv2.imshow("final",final)
    cv2.imwrite(dst + "/"+name + "_" + stage + "_5_CH+BB.jpg",final)
    
    
    return (center,dh,dv,candidato)
    
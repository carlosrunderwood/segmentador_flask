import cv2

import app.fuente.preprocesamiento as pre
import app.fuente.bib as bib
#import Ext_Disco_Optico as ExCC

#---------------------------------------------------------------
#   MENU
#---------------------------------------------------------------
def preprocesamiento(src,imgName,maskC,dst):
    stage = "Pre"
    img = pre.corte(src)
    maskC = pre.corte(maskC)
    
    clahe = bib.enhContrast(img, 3, 0)
    
    imgB, imgG, imgR = pre.BGR(clahe)
    
    homo = pre.Homomorfico(imgG,150)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_1_Homomorfico.jpg", homo)
    
    bh = pre.BlackHat(homo, 40, 40)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_2_BlackHat.jpg", bh)
    
    med = pre.Mediana(bh)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_3_Mediana.jpg",med)
    
    thresh = bib.umbralizar(med, 91)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_4_Adthresh.jpg",thresh)
    
    cierre = pre.Cierre(thresh,11)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_5_Cierre.jpg",cierre)
    
    dstG, dstB, dstR, dstT, repaint = pre.Repaint(img, cierre)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_6_Repintado.jpg",repaint)
    return (img,repaint,cierre,maskC)

def preprocesamientoSD(src,imgName,dst):
    stage = "Pre"
    img = pre.corte(src)
    
    clahe = bib.enhContrast(img, 3, 0)
    
    imgB, imgG, imgR = pre.BGR(clahe)
    
    homo = pre.Homomorfico(imgG,150)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_1_Homomorfico.jpg", homo)
    
    bh = pre.BlackHat(homo, 40, 40)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_2_BlackHat.jpg", bh)
    
    med = pre.Mediana(bh)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_3_Mediana.jpg",med)
    
    thresh = bib.umbralizar(med, 91)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_4_Adthresh.jpg",thresh)
    
    cierre = pre.Cierre(thresh,11)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_5_Cierre.jpg",cierre)
    
    dstG, dstB, dstR, dstT, repaint = pre.Repaint(img, cierre)
    
    cv2.imwrite(dst + "/"+imgName + "_" + stage + "_6_Repintado.jpg",repaint)
    return (img,repaint,cierre)



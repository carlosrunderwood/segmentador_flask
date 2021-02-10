import cv2 as cv2
import app.fuente.bib as bib


def copaSeg(src,img_name,dst):
    stage = "OC-seg"
    img = bib.corte(src)
    
    size = (img.shape[1]//2,img.shape[0]//2)
    
    resized = bib.resize(img,size)
    prep = bib.enhContrast(resized,3,0)
    
    shape = prep.shape
    
    azul, rojo, verde = bib.dividirCanales(prep)
    
    canal = azul
    
    cv2.imwrite(dst+"/"+img_name + "_" + stage + "_1_rgb-b.jpg",canal)
    
    homo = bib.Homomorfico(canal,70)
    cv2.imwrite(dst+"/"+img_name + "_" + stage + "_2_Homomorfico.jpg",homo)
    
    umbralizado = bib.umbralizar(homo,99)
    # print (umbral)
    cv2.imwrite(dst+"/"+img_name + "_" + stage + "_3_gthreshold.jpg",umbralizado)
    
    minimo = bib.minimo(umbralizado,3)
    cv2.imwrite(dst+"/"+img_name + "_" + stage + "_4_min.jpg",minimo)
    
    cerradura = bib.cierre(minimo,87)
    cv2.imwrite(dst+"/"+img_name + "_" + stage + "_5_Close.jpg",cerradura)
    
    candidato = bib.connectedComp(cerradura)
    
    contornos,indice = bib.bordesCanny(candidato)
    
    # convex = bib.Convex_Hull(contornos,shape,indice)
    
    elipse,filledElipse,(center,(dh,dv),ang) = bib.minEllipse(contornos, shape, indice)
    
    contornosE, indiceE = bib.bordesCanny(elipse)
    
    center = bib.float2int(center)
    final = bib.suma(prep,filledElipse)
    
    # cv2.imshow("final",final)
    cv2.imwrite(dst+"/"+img_name + "_" + stage + "_6_CH+BB.jpg",final)
    
    #resizedmask = bib.resize(mask, size)
    
    #dice = bib.diceCoefficient(filledElipse,resizedmask)
    
    #dif = cv2.subtract(cv2.bitwise_or(filledElipse,resizedmask), cv2.bitwise_and(filledElipse,resizedmask))
    
    #cv2.imwrite(dst+"/"+img_name + "_" + stage + "maskDifferences.jpg",dif)
    
    pixels = bib.ellipseArea(dh, dv)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return (center,dh,dv,pixels,elipse,prep)
import cv2 as cv2
import app.fuente.bib as bib


def omitir(src,name):
    w,h = src.shape[:2]
    if( (w < h/2) or (h < w/2) ):
        print("omitiendo la segmentacion de copa optica de: " + name)
        return True
    return False


def copaSeg(src,img_name,maskC,maskD,dst):
        stage = "OC-seg"

    
        if( omitir(src,img_name) ):
            return None,None,None,None,None,None,0,True
        
        
        img = bib.applyMask(src,maskD[:,:,0])
        mask = maskC
        
        
        prep = bib.enhContrast(img,3,0)
        
        shape = prep.shape
        
        azul, rojo, verde = bib.dividirCanales(prep)
        
        canal = azul
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_1_rgb-b.jpg",canal)
        
        homo = bib.Homomorfico(canal,80)
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_2_Homomorfico.jpg",homo)
        
        umbralizado = bib.umbralizar(homo,98.5)
        # print (umbral)
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_3_gthreshold.jpg",umbralizado)
        
        minimo = bib.minimo(umbralizado,11)
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_4_min.jpg",minimo)
        
        cerradura = bib.cierre(minimo,150)
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_5_Close.jpg",cerradura)
        
        candidato = bib.connectedComp(cerradura)
        
        finalMask = bib.cierre(candidato,400)
        
        contornos,indice = bib.bordesCanny(finalMask)
        
        cont = bib.drawContours(contornos, shape, indice)
        
        _,dh,dv,center= bib.boundingBox(contornos, shape, indice)
        
        center = bib.float2int(center)
        final = bib.suma(prep,cont)
        
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_6_CH+BB.jpg",final)
        
        finalMask = [finalMask,finalMask,finalMask]
        finalMask = cv2.merge(finalMask)
        
        dice = bib.diceCoefficient(finalMask,mask)
        
        dif = cv2.subtract(cv2.bitwise_or(finalMask,mask), cv2.bitwise_and(finalMask,mask))
        
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "maskDifferences.jpg",dif)
        
        
        copa, fondo = bib.contrast(src,mask)

        contraste = (copa - fondo) / (copa + fondo)
        print(contraste)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return (center,dh,dv,cont,prep,dice,False)

def copaSegSD(src,img_name,maskD,dst):
        stage = "OC-seg"

    
        if( omitir(src,img_name) ):
            return None,None,None,None,None,None,0,True
        
        
        img = bib.applyMask(src,maskD[:,:,0])
        
        
        
        prep = bib.enhContrast(img,3,0)
        
        shape = prep.shape
        
        azul, rojo, verde = bib.dividirCanales(prep)
        
        canal = azul
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_1_rgb-b.jpg",canal)
        
        homo = bib.Homomorfico(canal,80)
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_2_Homomorfico.jpg",homo)
        
        umbralizado = bib.umbralizar(homo,98.5)
        # print (umbral)
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_3_gthreshold.jpg",umbralizado)
        
        minimo = bib.minimo(umbralizado,11)
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_4_min.jpg",minimo)
        
        cerradura = bib.cierre(minimo,150)
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_5_Close.jpg",cerradura)
        
        candidato = bib.connectedComp(cerradura)
        
        finalMask = bib.cierre(candidato,400)
        
        contornos,indice = bib.bordesCanny(finalMask)
        
        cont = bib.drawContours(contornos, shape, indice)
        
        _,dh,dv,center = bib.boundingBox(contornos, shape, indice)
        
        center = bib.float2int(center)
        final = bib.suma(prep,cont)
        
        cv2.imwrite(dst+"/"+img_name + "_" + stage + "_6_CH+BB.jpg",final)
        
        finalMask = [finalMask,finalMask,finalMask]
        finalMask = cv2.merge(finalMask)
        
        return (center,dh,dv,cont,prep,False)
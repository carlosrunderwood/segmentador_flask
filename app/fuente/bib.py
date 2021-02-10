import cv2 as cv2
import numpy as np
import scipy.fftpack
import math

# Divide en canales la imagen original
def dividirCanales(src):
    h,w,c = src.shape
    (B,G,R) = cv2.split(src)
    return B, R, G

def percentil(src,p):
    per = np.percentile(src,p)
    return per

# Umbralizar la imagen
def umbralizar(src,parce):
    per = percentil(src,parce)
    ret,th1 = cv2.threshold(src,per,255,cv2.THRESH_BINARY)
    return (th1)

# Calcula el filtro minimo
def minimo(src,k):
    filterSize =(k,k) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   filterSize)
    minim = cv2.erode(src,kernel)
    return minim

# Cerradura binaria
def cierre(src, k):
    filterSize =(k, k) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize) 
    dil = cv2.dilate(src,kernel)
    ero = cv2.erode(dil,kernel)
    return ero

# Calculo de bordes
def bordesCanny(src):
    canny = cv2.Canny(src, 50, 150)
    # Buscamos los contornos
    (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    indice = 0
    i = 0
    mayor = 0
    for objetos in contornos:
        tamaño = len(objetos)
        # print("Tamaño: ",tamaño,"mayor: ",mayor)
        mayor = np.max([mayor,tamaño])
        if (tamaño == mayor):
            indice = i
        i += 1
    return (contornos,indice)

# Convex Hull
def Convex_Hull(contours,shape,indice):
    hulls = []
    hull = cv2.convexHull(contours[indice])
    hulls.append(hull)
    drawing = np.zeros(shape,dtype=np.uint8)
    color = (255,255,255)
    cv2.drawContours(drawing, hulls, 0, color)
    return drawing

#MinimumEllipseFit
def minEllipse(contours,shape,indice):
    ellipse = cv2.fitEllipse(contours[indice])
    drawing = np.zeros(shape,dtype=np.uint8)
    filledDrawing = np.zeros(shape,dtype=np.uint8)
    color = (255,255,255)
    cv2.ellipse(drawing, ellipse, color,2)
    cv2.ellipse(filledDrawing, ellipse, color,-1)
    return drawing, filledDrawing, ellipse

# bounding box
def boundingBox(src,contours,shape,indice):
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    c = contours[indice]
    contours_poly[0] = cv2.approxPolyDP(c, 3, True)
    boundRect[0] = cv2.boundingRect(contours_poly[0])
    drawing = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    puntos = boundRect[0]
    punto1 = (puntos[0],puntos[1])
    punto2 = (puntos[0]+puntos[2],puntos[1]+puntos[3])
    dh = punto2[0]-punto1[0]
    dv = punto2[1]-punto1[1]
    cv2.rectangle(drawing,punto1,punto2,(0,0,255),2)
    center = [punto1[0]+math.floor(dv/2.0),punto1[1]+math.floor(dh/2.0)]
    return (drawing,dv,dh,center)    
    
# suma dos imagenes
def suma(src1,src2):
    if(src1.shape != src2.shape):
        return []
    suma = cv2.add(src1,src2)
    return suma

# Rellenado de hoyos
def rellenado(X0,X1,Ac,B,primer):
	comp = X0 == X1
	# print(comp.all())
	if primer == 1 and comp.all():
		return X0
	X0 = X1
	X1 = cv2.dilate(X0,B)
	X1 = cv2.bitwise_and(X1,Ac)
	# cv2.imshow("X1",X1)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return rellenado(X0,X1,Ac,B,1)


def iterador(src,shape,k,hoyo):
    X = np.zeros(shape,dtype=np.uint8)
    B = cv2.getStructuringElement(cv2.MORPH_CROSS,(k,k))
    Ac = cv2.bitwise_not(src)
    X[hoyo[1]][hoyo[0]] = [255,255,255]
    X = rellenado(X,X,Ac,B,0)
    res = cv2.add(X,src)
    return res

def flood_Fill (reg, x, y):
    pixels = 0
    base, height, _ = reg.shape
    
    # Add (x,y) coordinates to set
    toFill = set()
    toFill.add((y,x))
    # print(reg[x][y])
    
    # Pops coordinates until 
    while True:
        # Pops from set, might empty it
        try:
            (a,b) = toFill.pop()
        except:
            # print("Flood Fill complete!")
            
            return pixels
        
        # Stop condition, this is how the set gets empty 
        if not np.allclose(reg[b][a],[0,0,0]):
            # Turns pixel to black
            reg[b][a] = (0,0,0)
            
            # Checks if pixel is a vein so and increases the count if so
            pixels += 1
            
            # Adds pixels in all directions, these might meet stop condition
            if a > 0:
                toFill.add((a-1,b))
            if a < base-1:
                toFill.add((a+1,b))
            if b > 0:
                toFill.add((a,b-1))
            if b < height-1:
                toFill.add((a,b+1))
            
    return pixels

def ExComponentesConectados(X0, X1, A, B, base):
	comparar = X0 == X1

	if base == 1 and comparar.all():
		return X0

	X0 = X1
	X1 = cv2.dilate(X0,B)
	X1 = cv2.bitwise_and(X1,A)

	return ExComponentesConectados(X0, X1, A, B, 1)

def ExInicio(src,shape,k,punto):
    X = np.zeros(shape, dtype = np.uint8)
    B = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    A = src
    X[punto[0]][punto[1]] = [255,255,255]
    X = ExComponentesConectados(X, X, A, B, 0)
    return X


def boundingEllipse(src,contours,shape,indice):
    contours_poly = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    c = contours[indice]
    contours_poly[0] = cv2.approxPolyDP(c, 3, True)
    centers[0], radius[0] = cv2.minEnclosingCircle(contours_poly[0])
    drawing = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    puntos = centers[0]
    punto1 = puntos[0]
    punto2 = puntos[1]
    cv2.circle(drawing, (int(punto1), int(punto2)), int(radius[0]), (255,0,0), 2)
    return (drawing,puntos)   

def float2int(array):
    array = np.round(array,decimals=0,out=None)
    for elem in array:
        elem = int(elem)
    return array.astype(int)

def ellipseArea(a,b):
    return math.pi * a * b

def diceCoefficient(src1, src2):
    dice = np.sum(src1[src2 == [255,255,255]])*2.0 / (np.sum(src1)+np.sum(src2))
    return dice

def corte(src):
    shape = src.shape
    out = src[0:shape[0],0:math.floor(shape[1]/2)]
    return out

def Homomorfico(img, sigma_):
    #number of rows and colums
    rows = img.shape[0]
    cols = img.shape[1]

    #number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = rows
    N = cols

    sigma = sigma_ #50

    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)

    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma)) #1 / (1 + (gaussianNumerator / Do)**(2*n)) 

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = np.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))

    # Set scaling factors and add
    Iout = Ioutlow #Iouthigh

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")

    return Ihmf2

def resize(src,size):
    h, w = src.shape[:2]
    dst = cv2.resize(src,size,interpolation = cv2.INTER_LINEAR)
    return dst

def equalizeColor(src):
    B,R,G = dividirCanales(src);
    
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    
    res = [B,G,R]
    
    return cv2.merge(res)

def equalizeIntensity(src):
    ycrcb =  cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    res = cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR)
    return res

def clahe(src,cp,add):
    clahe = cv2.createCLAHE(clipLimit=cp)
    res = clahe.apply(src) + add
    return res

def enhContrast(src,cp,add):
    B,R,G = dividirCanales(src);
    
    B = clahe(B,cp,add)
    G = clahe(G,cp,add)
    R = clahe(R,cp,add)
    
    res = [B,G,R]
    
    return cv2.merge(res)

def enhContrast2(src,cp,add):
    ycrcb =  cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    channels[0] = clahe(channels[0],cp,add)
    cv2.merge(channels,ycrcb)
    res = cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR)
    
    return res


def negativo(src):
    return cv2.bitwise_not(src)

def connectedComp(src):
    res = []
    maxi = 0
    ret, labels = cv2.connectedComponents(src)
    for label in range(1,ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255
        wpix = np.sum(mask == 255)
        if(wpix>maxi):
            maxi = wpix
            res = mask[:]
            
    return res
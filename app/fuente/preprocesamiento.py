import cv2
import numpy as np
import scipy.fftpack
import math

#---------------------------------------------------------------
# Filtro Homomorfico
#---------------------------------------------------------------
def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

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
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = np.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = np.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

    # Set scaling factors and add
    Iout = Ioutlow #Iouthigh

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")

    return Ihmf2
#---------------------------------------------------------------
# Divide los canales de la imagen
#---------------------------------------------------------------
def BGR(img):
    (B, G, R) = cv2.split(img)

    return B, G, R

#---------------------------------------------------------------
# BlackHat
#---------------------------------------------------------------
def BlackHat(img, filX, filY):
    filterSize = (filX, filY) #40,40
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   filterSize) 

    input_image = img
    #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) 
  
    # Applying the Black-Hat operation 
    tophat_img = cv2.morphologyEx(input_image,  
                                  cv2.MORPH_BLACKHAT, 
                                  kernel)

    return tophat_img

#---------------------------------------------------------------
# Calcular la mediana de la imagen
#---------------------------------------------------------------
def Mediana(img):
    newimg = cv2.medianBlur(img, 13)

    return newimg

#---------------------------------------------------------------
# Umbralizacion Global
#---------------------------------------------------------------
def pisodata(img,m,e,md):
    # print(m,md,e,m-md)
    if (np.abs(m-md)<e): 
        md = int(math.floor(md))
        return md
    
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    intm = int(math.floor(m))
    
    A = hist[:intm]
    B = hist[intm+1:]
    
    suma = 0
    for i in range(np.size(A)):
        suma += i*A[i]
    ma = suma / np.sum(A)
    suma = 0
    for i in range(np.size(B)):
        suma += i*B[i]
    mb = suma / np.sum(B)
    newm = (mb+ma)/2
    return pisodata(img,newm,e,m)

def Adthresh(img):
    m = int(math.floor(np.mean(img)))
    m = pisodata(img,m,0.0001,0)

    ret,th1 = cv2.threshold(img,m,255,cv2.THRESH_BINARY)

    return th1

#---------------------------------------------------------------
# Aplicacion de cierre a la imagen
#---------------------------------------------------------------
def Cierre(img, k):
    filterSize = (k,k) #7,7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize) 

    dil = cv2.dilate(img,kernel)

    ero = cv2.erode(dil,kernel)

    return ero

#---------------------------------------------------------------
# Repaint
#---------------------------------------------------------------
def Repaint(img, mascara):
    b, g, r = cv2.split(img)

    # print("Calculando la primera substraccion")
    dstG = cv2.inpaint(g,mascara,1,cv2.INPAINT_TELEA)
    # print("Calculando la segunda substraccion")
    dstB = cv2.inpaint(b,mascara,1,cv2.INPAINT_TELEA)
    # print("Calculando la tercera substraccion")
    dstR = cv2.inpaint(r,mascara,1,cv2.INPAINT_TELEA)

    # print("Calculando la imagen completa")

    dstT = np.zeros((img.shape))
    dstT[:,:,0] = dstG
    dstT[:,:,1] = dstB
    dstT[:,:,2] = dstR

    IMF2 = cv2.merge([dstB,dstG,dstR])

    return dstG, dstB, dstR, dstT, IMF2

def robustecer(src, k, l):
    blur = cv2.GaussianBlur(src,(l,l),0)
    filterSize =(k,k) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize) 
    ero = cv2.erode(blur,kernel)
    return ero

def percentil(src,p):
    per = np.percentile(src,p)
    return per

def Global(src,p):
    per = percentil(src, p)
    ret1,th1 = cv2.threshold(src,per,255,cv2.THRESH_BINARY)
    return th1

def Apertura(img, k):
    filterSize = (k, k) #7,7
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize) 
    ero = cv2.erode(img,kernel)
    dil = cv2.dilate(ero,kernel)
    return dil

def corte(src):
    shape = src.shape
    out = src[0:shape[0],0:math.floor(shape[1]/2)]
    return out

def otsuT(src):
    ret3,th3 = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

def threshold(src,t):
    ret,th1 = cv2.threshold(src,t,255,cv2.THRESH_BINARY)
    return th1
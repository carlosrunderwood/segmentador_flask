from flask import *
import cv2
from app import caracteristicas,comprime, app, caracteristicasSD


@app.route("/")
def upload():
    return render_template("subirImagen.html")

@app.route("/success",methods=["POST"])
def success():
    f=request.files['file']
    success.file_name=f.filename
    f.save("app/static/imagenes/"+success.file_name)
    img = cv2.imread("app/static/imagenes/"+success.file_name)
    if(request.form.get('calcular_dice')):
        m=request.files['file2']
        if(m != None):
            success.file_name_mascara = m.filename
            m.save("app/static/imagenes/"+success.file_name_mascara)
            imgM = cv2.imread("app/static/imagenes/"+success.file_name_mascara)
            name = success.file_name.split(sep=".")
            nameM = success.file_name_mascara.split(sep=".")
            dice = caracteristicas(img,success.file_name,imgM)
            comprime(name[0])
            return render_template("dice.html",name = name[0],name_file=success.file_name,nameMascara_file = success.file_name_mascara, dice = dice)
        else:
            return render_template("subirImagen.html")
    caracteristicasSD(img,success.file_name)
    name = success.file_name.split(sep=".")
    comprime(name[0])
    return render_template("exito.html",name=name[0])

@app.route("/download",methods=["POST"])
def download():
    filename=request.form.get("name")+".zip"
    return send_file("static/imagenes/"+filename,as_attachment=True)

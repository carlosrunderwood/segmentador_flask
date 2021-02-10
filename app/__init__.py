from flask import *
from app.fuente.cdr import caracteristicas
from app.fuente.comprimir import comprime
import cv2

app = Flask(__name__)
app.config.from_object("config.ProductionConfig")
from app.routes import *
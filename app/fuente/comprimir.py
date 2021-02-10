import os
import zipfile
 

def comprime(name):
    fantasy_zip = zipfile.ZipFile('app/static/imagenes/'+name+'.zip', 'w')
    
    for folder, subfolders, files in os.walk('app/static/imagenes/'+name):
        for file in files:
            fantasy_zip.write(os.path.join(folder, file),file, compress_type = zipfile.ZIP_DEFLATED)
    
    fantasy_zip.close()
import sys
import argparse
import os
import cdr

parser = argparse.ArgumentParser(description='Optic Disk and Cup segmentation.')

# Optional arguments definition
parser.add_argument('-d', '--directory', type=str, metavar='', required=True, help='Directory to fetch images from - i.e. "Healthy", "Glaucoma and suspects".')
parser.add_argument('-s', '--split', type=str, metavar='', required=False, help='Use only if images are split in a subdirectory - e.g. "Maci", "Neko".')

# Optional arguments for clarity
#group = parser.add_mutually_exclusive_group()
#group.add_argument('-q', '--quiet', action='store_true', help='Print quiet')
#group.add_argument('-v', '--verbose', action='store_true', help='Print verbose')

# Argument parsing
args = parser.parse_args()

def main():
    directory_path = ''

    try:
        subdirectory = args.split
        directory_path = 'RIM-ONE r3/' + args.directory + '/Stereo Images/' + subdirectory
    except:
        directory_path = 'RIM-ONE r3/' + args.directory + '/Stereo Images'

    expert_path = 'RIM-ONE r3/' + args.directory + '/Expert1_masks'
    
    # Get all files and subdirectories
    file_list = os.listdir(directory_path)
    
    avg = 0.0
    avg0 = 0.0
    
    count = 0
    count0 = 0
    
    maxC = 0.0
    imMaxC = ""
    minC = 1.0
    imMinC = ""
    avgC = 0.0

    # Iterates through list
    for file in file_list:
        # Opens only files
        if os.path.isfile(os.path.join(os.path.join(os.getcwd(),directory_path),file)):
            
            # Gets file without extension
            filename = os.path.splitext(file)[0]
            print ('\n\nNow inspecting "' + file + '".')
            
            try:
                # Tries to create a directory for all new images, shares name with original file
                try:
                    os.mkdir(os.path.join(directory_path, filename))  
                    
                except:  
                    pass 
                
                # All the magic happens here
                dice, contraste = cdr.caracteristicas(expert_path, directory_path, filename, os.path.join(directory_path, filename))
                
                if dice > 0.0:
                    avg += dice
                    count += 1
                
                avg0 += dice
                count0 += 1

                if contraste > maxC:
                    maxC = contraste
                    imMaxC = file
                elif contraste < minC:
                    minC = contraste
                    imMinC = file

                avgC += contraste
            
            except:
                print ('There was a problem with "'+ file + '".')
    

    print ("\nAverage: ", avg/count)
    print ("Count: ", count)
    
    print ("Average at 0: ", avg0/count0)
    print ("Count at 0: ", count0)

    print("Max Contraste: "+str(maxC)+" imagen: "+imMaxC)
    print("Min Contraste: "+str(minC)+" imagen: "+imMinC)
    print("Contraste medio: "+str(avgC/count0))

if __name__ == "__main__":
    main()

import os
import re

# folders = ['FASTEST_1', 'FASTEST_2', 'FASTEST_3', 'FASTEST_4']
folders = ['FASTEST_1']

for folder in folders:
    print(f'Getting {folder}')

    # Mit res

    # Oszillierende Blase
    os.system(f'rsync -vLa --delete-before --delete-excluded --include="res/***" --include="out/***" --include="id/***" --include="map/***" --include="output.txt" --include="funcprp.F" --include="funcusr.F" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude="out/oscillation.res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{folder}/projects/02_Oscillation/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

    # Aufsteigende Blase
    # os.system(f'rsync -vLa --delete-before --delete-excluded --include="res/***" --include="out/***" --include="id/***" --include="map/***" --include="output.txt" --include="funcprp.F" --include="funcusr.F" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude="out/oscillation.res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{folder}/projects/rbubble_test1/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

    # Static Bubble mit output.txt
    # os.system(f'rsync -vLa --delete-before --delete-excluded --include="res/***" --include="out/***" --include="id/***" --include="map/***" --include="funcprp.F" --include="funcusr.F" --include="output.txt" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude="out/oscillation.res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{folder}/projects/01_StaticBubble/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

    # Ohne output.txt
    # os.system(f'rsync -vLa --delete-before --delete-excluded --include="res/***" --include="out/***" --include="id/***" --include="map/***" --include="funcprp.F" --include="funcusr.F"  --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude="out/oscillation.res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{folder}/projects/01_StaticBubble/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

    # Oszillierende Blase ohne res

    # os.system(f'rsync -vLa --delete-before --delete-excluded --include="out/***" --include="id/***" --include="map/***" --include="funcprp.F" --include="funcusr.F" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude=".res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{folder}/projects/02_Oscillation/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

    # Oszillierende Blase mit output.txt
    # os.system(f'rsync -vLa --delete-before --delete-excluded --exclude=".res" --include="out/***" --include="id/***" --include="output.txt" --include="map/***" --include="funcprp.F" --include="funcusr.F" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude="out/oscillation.res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{folder}/projects/02_Oscillation/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

print('Executing regex')

# Convert y_pos.txt to csv and store in y_pos_re.txt
for path in folders:
    if os.path.isfile('./'+path+'/y_pos.txt'):
        with open('./'+path+'/y_pos.txt', 'r') as myfile:
            data = myfile.read()
            data = re.sub(r'(\d) +(\d)', r'\1, \2', data)
            data = re.sub(r'(?<!,) +', r'', data)

            outfile = open('./'+path+'/y_pos_re.txt', 'w')
            outfile.write(data)
            outfile.close()

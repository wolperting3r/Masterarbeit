import os
import re

# folders = ['FASTEST_1', 'FASTEST_2', 'FASTEST_3', 'FASTEST_4']
# folders = ['FASTEST_1', 'FASTEST_2', 'FASTEST_3']
# folders = ['FASTEST_3', 'FASTEST_4']
# folders = ['FASTEST_1', 'FASTEST_3']
folders = ['FASTEST_1']

for folder in folders:
    # if folder == 'FASTEST_1':
        # maira_folder = 'FASTEST'
    # else:
    maira_folder = folder

    print(f'Getting {folder}')
    # Mit res
    # if ('3' in folder) or ('4' in folder):
    os.system(f'rsync -vLa --delete-before --delete-excluded --include="res/***" --include="out/***" --include="id/***" --include="map/***" --include="funcprp.F" --include="funcusr.F" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude="out/oscillation.res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{maira_folder}/projects/02_Oscillation/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')
    # Static Bubble mit output.txt
    # os.system(f'rsync -vLa --delete-before --delete-excluded --include="res/***" --include="out/***" --include="id/***" --include="map/***" --include="funcprp.F" --include="funcusr.F" --include="output.txt" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude="out/oscillation.res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{maira_folder}/projects/01_StaticBubble/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')
    # Ohne output.txt
    # os.system(f'rsync -vLa --delete-before --delete-excluded --include="res/***" --include="out/***" --include="id/***" --include="map/***" --include="funcprp.F" --include="funcusr.F"  --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude="out/oscillation.res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{maira_folder}/projects/01_StaticBubble/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

    # Ohne res
    # if ('2' in folder) or ('1' in folder):
    # os.system(f'rsync -vLa --delete-before --delete-excluded --include="out/***" --include="id/***" --include="map/***" --include="funcprp.F" --include="funcusr.F" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" --exclude=".res" --exclude=".map" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{maira_folder}/projects/02_Oscillation/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')
    # os.system(f'rsync -va --delete-before zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{maira_folder}/projects/02_Oscillation/src/fhp/fsrc/curv_ml.F /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

# '''
print('Executing regex')

for path in folders:
    with open('./'+path+'/y_pos.txt', 'r') as myfile:
        data = myfile.read()
        data = re.sub(r'(\d) +(\d)', r'\1, \2', data)
        data = re.sub(r'(?<!,) +', r'', data)

        outfile = open('./'+path+'/y_pos_re.txt', 'w')
        outfile.write(data)
        outfile.close()
# '''

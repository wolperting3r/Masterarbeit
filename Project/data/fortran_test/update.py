import os
import re

# folders = ['FASTEST_1', 'FASTEST_2', 'FASTEST_3', 'FASTEST_4']
folders = ['FASTEST_1', 'FASTEST_2']

for folder in folders:
    if folder == 'FASTEST_1':
        maira_folder = 'FASTEST'
    else:
        maira_folder = folder

    print(f'Getting {folder}')
    os.system(f'rsync -vLa --delete-before --include="res/***" --include="out/***" --include="id/***" --include="map/***" --include="funcusr.F" --include="curv_ml.F" --include="y_pos.txt" --exclude="*" zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{maira_folder}/projects/02_Oscillation/ /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')
    # os.system(f'rsync -va --delete-before zkraus@Maira.fnb.maschinenbau.tu-darmstadt.de:/work/local/zkraus/{maira_folder}/projects/02_Oscillation/src/fhp/fsrc/curv_ml.F /Users/zach/Git/Masterarbeit/Project/data/fortran_test/{folder}/')

print('Executing regex')

for path in folders:
    with open('./'+path+'/y_pos.txt', 'r') as myfile:
        data = myfile.read()
        data = re.sub(r'(\d) +(\d)', r'\1, \2', data)
        data = re.sub(r'(?<!,) +', r'', data)

        outfile = open('./'+path+'/y_pos_re.txt', 'w')
        outfile.write(data)
        outfile.close()

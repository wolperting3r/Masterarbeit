import re

with open('./2006011726 Stabiles Modell dshift1b/y_pos.txt', 'r') as myfile:
    data = myfile.read()
    data = re.sub(r'(\d) +(\d)', r'\1, \2', data)
    data = re.sub(r'(?<!,) +', r'', data)

    outfile = open('./2006011726 Stabiles Modell dshift1b/y_pos_re.txt', 'w')
    outfile.write(data)
    outfile.close()

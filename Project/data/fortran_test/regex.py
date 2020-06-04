import re

# path = 'FASTEST_2'
# paths = ['FASTEST_1', 'FASTEST_3']
path = ['FASTEST_1', 'FASTEST_2', 'FASTEST_3', 'FASTEST_4']
for path in paths:
    with open('./'+path+'/y_pos.txt', 'r') as myfile:
        data = myfile.read()
        data = re.sub(r'(\d) +(\d)', r'\1, \2', data)
        data = re.sub(r'(?<!,) +', r'', data)

        outfile = open('./'+path+'/y_pos_re.txt', 'w')
        outfile.write(data)
        outfile.close()

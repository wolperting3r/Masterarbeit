import re

# path = 'FASTEST_2'
# paths = ['FASTEST_1', 'FASTEST_3']
paths = [
    '2006200857 cvofls Netz eqk',
    '2006200857 cvofls Netz',
]
for path in paths:
    with open('./'+path+'/y_pos.txt', 'r') as myfile:
        data = myfile.read()
        data = re.sub(r'(\d) +(\d)', r'\1, \2', data)
        data = re.sub(r'(?<!,) +', r'', data)

        outfile = open('./'+path+'/y_pos_re.txt', 'w')
        outfile.write(data)
        outfile.close()

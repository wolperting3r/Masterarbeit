import re

with open('output.txt', 'r') as myfile:
    data = myfile.read()
    data = re.sub(r' +', '', data)
    data = re.sub(r'^[\d\D]+?\+-------------------------------------------\+\n(c\n)', r'\1', data)
    data = re.sub(r'(\d)\n(\d)', r'\1, \2', data)
    data = re.sub(r'\+-------------------------------------------\+\n.*\n\+-------------------------------------------\+\n\n(.*\n)+?\+-------------------------------------------\+\n.*\n\+-------------------------------------------\+\n', r'', data)

    data_c = re.sub(r'cm\n.*\n', r'', data)
    data_c = re.sub(r'last.*\n?', r'', data_c)
    data_c = re.sub(r'c\n', r'', data_c)
    data_c = 'c\n' + data_c
    last_c = open('output_c.txt', 'w')
    last_c.write(data_c)
    last_c.close()

    data_cm = re.sub(r'c\n.*\n', r'', data)
    data_cm = re.sub(r'last.*\n?', r'', data_cm)
    data_cm = re.sub(r'cm\n', r'', data_cm)
    data_cm = 'cm\n' + data_cm
    last_cm = open('output_cm.txt', 'w')
    last_cm.write(data_cm)
    last_cm.close()


    data_last = re.sub(r'c.*\n.*\n', r'', data)
    data_last = re.sub(r'lastoutput', r'', data_last)
    data_last = 'last\n' + data_last
    last_file = open('output_last.txt', 'w')
    last_file.write(data_last)
    last_file.close()

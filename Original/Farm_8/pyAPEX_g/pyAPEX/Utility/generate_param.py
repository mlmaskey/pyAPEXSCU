# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:00:41 2022

@author: Mahesh.Maskey
"""


file_path = "APEXPARM.DAT"
# read text file output from APEX
with open(file_path) as f:
    # ref: https://www.delftstack.com/howto/python/python-readlines-without-newline/
    lines=f.read().splitlines()
f.close()


f_w = open("simAPEXPARM.DAT", "w")
f_w.close()
f_a = open("simAPEXPARM.DAT", "a")    
# lines 1 to 35
for i in range (0, 35):
    lines_read = list(lines[i].split(' '))
    n_text = len(lines_read)
    idxes = []
    for j in range(n_text):
        if lines_read[j] != '':
            print(lines_read[j])
            idxes.append(j)
    scrp1, scrp2 = float(lines_read[idxes[0]]), float(lines_read[idxes[1]])            
    lines_write = lines_read
    lines_write[idxes[0]] = str("%.2f" % scrp1)
    lines_write[idxes[1]] = str("%.2f" % scrp2)+'\n'
    lines_write = ' '.join(lines_write)    
    f_a.writelines(lines_write)
# lines 36 to 45
lines_read = list(lines[35].split(' '))
for p in range(2, 11):
    n_i = i+p
    lines_read = list(lines[n_i].split(' '))
    n_text = len(lines_read)
    idxes = []
    for j in range(n_text):
        if lines_read[j] != '':
            print(lines_read[j])
            idxes.append(j)
    lines_write = lines_read
    for idx in idxes:
        lines_write[idx] = str("%.1f" % float(lines_read[idx]))
    lines_write[-1] = lines_write[-1]+'\n'
    lines_write = ' '.join(lines_write)
    f_a.writelines(lines_write) 
lines_read = list(lines[n_i+1].split(' '))
n_text = len(lines_read)
idxes = []
for j in range(n_text):
    if lines_read[j] != '':
        print(lines_read[j])
        idxes.append(j)
lines_write = lines_read
for idx in idxes:
    lines_write[idx] = str("%.1f" % float(lines_read[idx]))
lines_write = ' '.join(lines_write)
f_a.writelines(lines_write)
f_a.close()

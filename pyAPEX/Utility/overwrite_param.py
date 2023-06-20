# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:13:04 2022

@author: Mahesh.Maskey
"""
import fortranformat as ff
from Utility.apex_utility import read_param_file
from Utility.apex_utility import write_line_ff
import numpy as np


def overwrite_param(file_read, file_write, p):
    print("Reading default APEX parameter file: " + str(file_read) + '\n')
    lines_param = read_param_file(file_read)

    # Create blank file
    f_w = open(file_write, "w")
    f_w.close()
    del f_w

    # from line 1 to 35
    for i in range(0, 35):
        # Open file to append
        f_a = open(file_write, "a")
        read_format = ff.FortranRecordReader('(2F8.2)')
        line_read = read_format.read(lines_param[i])
        # update the parameters
        line_read[0] = float(p[2 * i])
        line_read[1] = float(p[2 * i + 1])
        write_format = ff.FortranRecordWriter('(2F8.2)')
        line_write = write_format.write(line_read)
        f_a.writelines(line_write + '\n')
        f_a.close()
    del i, line_read, line_write, read_format, write_format, f_a

    inc_param = np.arange(70, 180, 10)
    id_lines = np.arange(35, 46)

    for idl in range(11):
        if idl == 10:
            write_line_ff(file_write, lines_param, id_lines[idl], p, inc_param[idl], nparam=4)
        else:
            write_line_ff(file_write, lines_param, id_lines[idl], p, inc_param[idl])

    print("Written new parameter into file: " + str(file_write))
    return p

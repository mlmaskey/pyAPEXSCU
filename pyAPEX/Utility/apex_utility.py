# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 09:52:40 2022

@author: Mahesh.Maskey
"""

import os
import random
import shutil
from datetime import date
import fortranformat as ff
import numpy as np
import pandas as pd
from Utility.easypy import easypy as ep


def nan_matrix(nr, nc):
    py_mat = np.zeros((nr, nc))
    py_mat[:] = np.nan
    return py_mat


def interpolate_param(mu, delta, diff, n, x):
    p = mu + delta * diff * 0.01
    if p < n:
        p = n
    elif p > x:
        p = x
    return p


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
		ref: https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
	"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def backup_program():
    # Copies the existing files into a Backup directory
    today = date.today()
    # Fetching all the files to directory
    bacup_directory = 'Backup_' + str(today)
    if not os.path.isdir(bacup_directory):
        shutil.copytree('Program', bacup_directory)
    else:
        shutil.rmtree(bacup_directory)
        shutil.copytree('Program', bacup_directory)
        print("File Copied Successfully")
    return


def get_range(file):
    # import csv file containing range of parameters with recommended values for specific project
    df_param_limit = pd.read_csv(file, index_col=0, encoding="ISO-8859-1")
    # ref: https://exerror.com/unicodedecodeerror-utf-8-codec-cant-decode-byte-0x96-in-position-35-invalid-start-byte/
    # ref: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    param_discription = df_param_limit.columns.to_list()
    param_name = df_param_limit.iloc[0, :].to_list()
    array_param_list = df_param_limit.iloc[1:, :].to_numpy()
    array_param_list = array_param_list.astype(np.float64)
    # ref: https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
    mat_param_list = np.asmatrix(array_param_list)
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html
    mat_param_list = np.squeeze(np.asarray(mat_param_list))
    # ref: https://stackoverflow.com/questions/3337301/numpy-matrix-to-array
    return mat_param_list, df_param_limit, param_discription, param_name


def read_param_file(file):
    with open(file) as f:
        # ref: https://www.delftstack.com/howto/python/python-readlines-without-newline/
        lines = f.read().splitlines()
    f.close()
    return lines


def line_2_items(lines, id_line, id_pos1, id_pos2, p1, p2, preci):
    line_read = list(lines[id_line].split(' '))
    line_write = line_read
    line_write[id_pos1] = str(preci % p1)
    line_write[id_pos2] = str(preci % p2) + '\n'
    line_write = ' '.join(line_write)
    return line_write


def modify_list(file, file_write):
    # Reads the APEXFILE.DAT to replace the APEX parameter file name
    print('Modifying the list of files in ' + str(file))
    file_list = read_param_file(file)
    f_w = open(file, "w")
    f_w.close()
    f_a = open(file, "a")
    lines_read = []
    for line in file_list:
        items = list(line.split())
        if items[0] == 'FPARM':
            items[1] = file_write.name
            line = ' ' + items[0] + '    ' + items[1]
            lines_read.append(line)
            f_a.writelines(line + '\n')
        else:
            lines_read.append(items)
            f_a.writelines(line + '\n')
    f_a.close()
    return


def write_line(file_write, lines, idline, params, id_start_param):
    lines_read = list(lines[idline].split(' '))
    n_text = len(lines_read)
    idxes = []
    for j in range(n_text):
        if lines_read[j] != '':
            idxes.append(j)
    idparams = range(id_start_param, id_start_param + len(idxes))
    for ip in range(len(idxes)):
        if idparams[ip] == 92 or idparams[ip] == 112 or idparams[ip] == 124 or idparams[ip] == 127 or idparams[
            ip] == 153 or idparams[ip] == 154:
            lines_read[idxes[ip]] = str("%.4f" % float(params[idparams[ip]]))
        else:
            lines_read[idxes[ip]] = str("%.3f" % float(params[idparams[ip]]))
    lines_write = ' '.join(lines_read)
    f_a = open(file_write, "a")
    f_a.writelines(lines_write + '\n')
    f_a.close()


def write_line_ff(file_write, lines, idl, params, start_id_param, nparam=10):
    if nparam == 4:
        read_format = ff.FortranRecordReader('(4F8.3)')
        line_read = read_format.read(lines[idl])
    else:
        read_format = ff.FortranRecordReader('(10F8.3)')
        line_read = read_format.read(lines[idl])
    for i in range(len(line_read)):
        line_read[i] = params[start_id_param + i]
    write_format = ff.FortranRecordWriter('(10F8.3)')
    items_list = []
    for item in line_read:
        items_list.append(float(item))
    line_write = write_format.write(items_list)
    # Open file to append
    f_a = open(file_write, "a")
    f_a.writelines(line_write + '\n')
    f_a.close()


def read_sensitive_params(src_dir):
    with open(src_dir / 'Utility/sensitive.PAR') as f:
        line = f.read()
    f.close()
    l = line.split(',')
    id_sensitive = [int(item) for item in l]
    for i in range(len(id_sensitive)):
        id_sensitive[i] = id_sensitive[i] + 69
    return id_sensitive


def generate_param_set(file_limits, n_sim=100, isall=True):
    mat_param_limt, df_limit, discription, param_list = get_range(file_limits)
    recc_params = mat_param_limt[2, :]
    if isall:
        min_params = mat_param_limt[0, :]
        max_params = mat_param_limt[1, :]
        n_params = len(max_params)
    else:
        id_sensitive = read_sensitive_params()
        mat_sensitive_limt = mat_param_limt[:, id_sensitive]
        min_params = mat_sensitive_limt[0, :]
        max_params = mat_sensitive_limt[1, :]
        recc_sensitive = mat_sensitive_limt[2, :]
        n_params = len(recc_sensitive)

    diff = max_params - min_params
    inc = diff / n_sim
    mat_params = np.zeros((n_sim + 1, n_params))
    if isall:
        mat_params[0, :] = recc_params
    else:
        mat_params[0, :] = recc_sensitive

    for i in range(n_sim):
        for j in range(n_params):
            mat_params[i + 1, j] = min_params[j] + i * inc[j]
    return recc_params, mat_params, param_list


def pick_param(mat_p, allparam=None):
    nset, nparam = mat_p.shape
    p = np.zeros((nparam))
    for ip in range(nparam):
        random.seed()
        id_rand = random.randint(0, nset - 1)
        p[ip] = mat_p[id_rand, ip]
    if (allparam is None):
        p = p
    else:
        id_sensitive = read_sensitive_params()
        allparam[id_sensitive] = p
        p = allparam
    return p


def get_control_period():
    # Extracts the simulation period, start date from APEXCONT.DAT file
    # and computes the date vector in three time scales: days, month, and year
    lines = read_param_file('APEXCONT.DAT')
    read_format = ff.FortranRecordReader('(20I6)')
    line_read = read_format.read(lines[0])
    n_years = line_read[0]
    startyear = line_read[1]
    startmonth = line_read[2]
    startday = line_read[3]
    lastyear = startyear + n_years - 1
    startdate = pd.to_datetime(str(startyear) + '/' + str(startmonth) + '/' + str(startday))
    stopdate = pd.to_datetime(str(lastyear) + '/12/31')
    vec_date = pd.date_range(startdate, stopdate, freq='d')
    vec_month = pd.date_range(startdate, stopdate, freq='M')
    vec_year = pd.date_range(startdate, stopdate, freq='Y')
    return (n_years, startyear, lastyear), (startdate, stopdate), (vec_year, vec_month, vec_date)


def txt2list(file):
    print(file)
    # read text file output from APEX
    with open(file, encoding="ISO-8859-1") as f:
        lines = f.readlines()
    f.close()
    # perse data from the text list above
    line_list = []
    for line in lines:
        l = list(line.split(' '))
        items = []
        for ele in l:
            if ele != '':
                items.append(ele)
        line_list.append(items)
    del line, lines
    return line_list


def get_daily_dwsv0(run_name):
    # read DWS file from APEX run

    file = run_name + '.DWS'

    line_list = txt2list(file)
    data_list = line_list[3:]
    header = line_list[2]
    # convert  list into dataframe
    df = pd.DataFrame(data_list)
    df.columns = header
    df.Y = df.Y.astype('int')
    df.M = df.M.astype('int')
    df.D = df.D.astype('int')
    for col in header[3:]:
        df[col] = df[col].astype('Float64')

    df.insert(3, "Date", df.Y.astype('str') + '/' + df.M.astype('str') + '/' + df.D.astype('str'))
    df.Date = pd.to_datetime(df.Date)
    df.index = df.Date
    df = df.drop(['Date'], axis=1)
    return df


def get_daily_dws(run_name):
    try:
        file = run_name + '.DWS'

        # line_list = txt2list(file)   
        # data_list = line_list[3:]
        # header = line_list[2]
        # df = pd.DataFrame (data_list)
        # df.columns = header
        # df1 =  df[header[:-12]]

        # df1.Y = df.Y
        # df1.M = df.M.astype('int')
        # df1.D = df.D.astype('int')
        # for col in df1.columns:
        #     df1[col] = df1[col].astype('Float64')

        # df1.insert(3, "Date",df.Y.astype('str')+'/'+df.M.astype('str')+'/'+df.D.astype('str'))
        # df1.Date =  pd.to_datetime(df1.Date)
        # df1.index =  df1.Date
        # df1 = df1.drop(['Date'], axis=1)

        df1 = pd.read_fwf(file, skiprows=2, encoding="ISO-8859-1", widths=[5, 4, 4] + [10] * 21)
        df1.insert(3, "Date", df1.Y.astype('str') + '/' + df1.M.astype('str') + '/' + df1.D.astype('str'))
        df1.Date = pd.to_datetime(df1.Date)
        df1.index = df1.Date
        df1 = df1.drop(['Date'], axis=1)

        return df1
    except:
        raise Exception('error occurs in get_daily_dws')


def get_daily_sadV0(run_name):
    # read SAD file from APEX run
    file = run_name + '.SAD'
    line_list = txt2list(file)
    data_list = line_list[3:]
    header = line_list[2]

    # convert  list into dataframe
    try:
        df = pd.DataFrame(data_list)
        df.columns = header
    except ValueError:
        return

    df['SA#'] = df['SA#'].astype('int')
    df.ID = df.ID.astype('int')
    df.Y = df.Y.astype('int')
    df.M = df.M.astype('int')
    df.D = df.D.astype('int')
    for col in header[6:]:
        try:
            df[col] = df[col].astype('Float64')
        except:
            continue
    df.insert(3, "Date", df.Y.astype('str') + '/' + df.M.astype('str') + '/' + df.D.astype('str'))
    df.Date = pd.to_datetime(df.Date)
    df.index = df.Date
    df = df.drop(['Date'], axis=1)
    return df


def get_daily_sad(run_name):
    try:
        # read SAD file from APEX run
        file = run_name + '.SAD'

        # line_list = txt2list(file)   
        # data_list = line_list[3:]
        # header = line_list[2]
        # header[-1] = 'Flag'
        # df = pd.DataFrame (data_list)
        # df.columns = header
        # ndata = df.shape[0]
        # df1 = df.copy()
        # for j in range(ndata):
        #     df_list = df1.iloc[j, :]
        #     item_list = df_list.to_list()
        #     nitems = len(item_list)
        #     item_list_new = []
        #     for i in range(nitems):
        #         try:
        #             item = item_list[i]
        #             # print(i, item)
        #             if ep.isfloat(item):
        #                 item_list_new.append(item)
        #             elif '*' in item:
        #                 item_list_new.append(np.nan)
        #                 item_list_new.append(np.nan)  
        #             elif 'NaN' in item:
        #                     item_list_new.append(item[0:3])
        #                     item_list_new.append(item[3:])                
        #             else:
        #                 if len(item) > 6:
        #                     item_list_new.append(item[0:6])
        #                     item_list_new.append(item[6:])
        #                 else:
        #                     item_list_new.append(item)
        #             # print(i, item_list_new[i])
        #         except:
        #             continue
        #     df1.iloc[j, :]=item_list_new
        # df1[df1=='NaN'] = np.nan

        df1 = pd.read_fwf(file, skiprows=2, encoding="ISO-8859-1", widths=[9, 8, 5, 4, 4] + [10] * 58)
        df1[df1 == 'NaN'] = np.nan
        df1.rename(columns={df1.columns[-1]: "Flag"}, inplace=True)

        df1['SA#'] = df1['SA#'].astype('int')
        df1.ID = df1.ID.astype('int')
        df1.Y = df1.Y.astype('int')
        df1.M = df1.M.astype('int')
        df1.D = df1.D.astype('int')

        for col in df1.columns[6:]:
            df1[col] = df1[col].astype('Float64')
        # for col in header[6:]:
        #     df1[col] = df1[col].astype('Float64')

        df1.insert(3, "Date", df1.Y.astype('str') + '/' + df1.M.astype('str') + '/' + df1.D.astype('str'))
        df1.Date = pd.to_datetime(df1.Date)
        df1.index = df1.Date
        df1 = df1.drop(['Date'], axis=1)
        return df1
    except:
        raise Exception('error in get_daily_sad')


# def get_daily_dps(run_name):
#     # read DPS file from APEX run
#     file = run_name + '.DPS'
#     line_list = txt2list(file)
#     header = line_list[3][0:5] + line_list[2][0:4] + [line_list[3][7], 'PSTNSUB']+line_list[2][4:] 
#     nrecords = len(line_list)
#     data_list = line_list[5:nrecords:2]
#     df = pd.DataFrame (data_list)
#     df.columns = header
#     df = df[['Y', 'M', 'D', 'YSD']]
#     # df = df.drop(['PSTN', 'PSTNSUB'], axis=1)
#     df.Y = df.Y.astype('int')
#     df.M = df.M.astype('int')
#     df.D = df.D.astype('int')
#     for col in df.columns[5:]:
#         df[col] = df[col].astype('Float64')
#     df.insert(3, "Date",df.Y.astype('str')+'/'+df.M.astype('str')+'/'+df.D.astype('str'))
#     df.Date =  pd.to_datetime(df.Date)
#     df.index =  df.Date
#     df = df.drop(['Date'], axis=1)
#     return df
def get_daily_dps(run_name):
    try:
        # read DPS file from APEX run upto YSD
        file = run_name + '.DPS'

        # line_list = txt2list(file)
        # data_list = line_list[5:]
        # data_list_new = []
        # for i in range(len(data_list)):
        #     try: 
        #         x = float(data_list[i][5])
        #         y = data_list[i][0:9]
        #         data_list_new.append(y)
        #     except ValueError:
        #         x = np.nan
        #         del x
        # header = line_list[3][0:5] + line_list[2][0:4]         
        # df = pd.DataFrame (data_list_new)
        # df.columns = header

        h1 = pd.read_fwf(file, skiprows=3, encoding="ISO-8859-1", widths=[9, 8, 5, 3, 3], nrows=1)
        h2 = pd.read_fwf(file, skiprows=2, encoding="ISO-8859-1", colspecs=[(29, 37), (37, 47), (47, 57), (57, 67)],
                         nrows=1)
        df = pd.read_fwf(file, skiprows=5, encoding="ISO-8859-1", widths=[9, 8, 5, 3, 3, 11] + [10] * 3, \
                         header=None, names=h1.columns.to_list() + h2.columns.to_list()). \
            query('Q==Q')

        df.Y = df.Y.astype('int')
        df.M = df.M.astype('int')
        df.D = df.D.astype('int')
        for col in df.columns[5:]:
            df[col] = df[col].astype('Float64')
        df.insert(3, "Date", df.Y.astype('str') + '/' + df.M.astype('str') + '/' + df.D.astype('str'))
        df.Date = pd.to_datetime(df.Date)
        df.index = df.Date
        df = df.drop(['Date'], axis=1)

        return df
    except:
        raise Exception('error occurs in get_daily_dps')


def get_acy(run_name):
    try:
        file = run_name + '.ACY'

        # line_list = txt2list(file)    
        # data_list = line_list[3:]
        # header = line_list[2]
        # # convert  list into dataframe
        # df = pd.DataFrame (data_list)
        # df.columns = header
        # df['SA#'] = df['SA#'].astype('int')
        # df.ID = df.ID.astype('int')
        # df.YR = df.YR.astype('int')
        # df['YR#'] = df['YR#'].astype('int')
        # for col in header[4:]:
        #     try:
        #         df[col] = df[col].astype('Float64')
        #     except: 
        #         continue 

        df = pd.read_fwf(file, skiprows=2, encoding="ISO-8859-1", widths=[9, 8, 5, 5, 5] + [10] * 21)

        return df
    except:
        raise Exception('error occurs in get_acy')


def pbias(ox, sx):
    return np.sum(ox - sx) * 100 / np.sum(ox)


def rmse(ox, sx):
    RMSE = (np.sum((ox - sx) ** 2) / len(ox)) ** 0.5
    return RMSE


def nrmse(RMSE, ox):
    return RMSE / np.mean(ox)


def nash(ox, sx):
    omu = np.mean(ox)
    NSE = 1 - np.sum((ox - sx) ** 2) / np.sum((ox - omu) ** 2)
    return NSE


# def cod(ox, sx):
#     corr,_ = pearsonr(ox, sx)
#     COD = corr**2
#     return COD

def ioa(x: list, y: list):
    mu_x = np.mean(x)
    d = 1 - (np.sum((x - y) ** 2)) / np.sum((abs(y - mu_x) + abs(x - mu_x)) ** 2)
    return d


def obj_fun(nse, pbais, corr):
    of1 = ((1 - nse) ** 2 + (abs(pbais) + 1 / 2) ** 2) ** 0.5
    of2 = ((1 - nse) ** 2 + (1 - corr) ** 2 + (abs(pbais) + 1 / 3) ** 2) ** 0.5
    return of1, of2


def perf_eval(X, Y):
    ox, sx = np.array(X), np.array(Y)
    COD = ep.nancorr(ox, sx)
    RMSE = rmse(ox, sx)
    nRMSE = nrmse(RMSE, ox)
    NSE = nash(ox, sx)
    PBIAS = pbias(ox, sx)
    APBIAS = np.abs(PBIAS)
    d = ioa(ox, sx)
    of1, of2 = obj_fun(NSE, PBIAS, COD)
    print('COD       ' + 'RMSE      ' + 'nRMSE     ' + 'NSE       '  'PBIAS       '  'IOA       '  'OF1      '  'OF2')
    print(str('%0.4F' % COD) + ',   ' + str('%0.3F' % RMSE) + ',   ' + str('%0.3F' % nRMSE) +
          ',   ' + str('%0.3F' % NSE) + ',   ' + str('%0.2F' % PBIAS) + ',   ' + str('%0.3F' % d) + ',   ' + str(
        '%0.3F' % of1) + ',   ' + str('%0.3F' % of2))
    df = pd.DataFrame([COD, RMSE, nRMSE, NSE, APBIAS, PBIAS, d, of1, of2],
                      index=['COD', 'RMSE', 'nRMSE', 'NSE', 'APBIAS', 'PBIAS', 'd', 'of1', 'of2'])
    return df.T


def import_data(file_path, WA):
    '''    Imports calibration data 
    Parameter: Complete file path of observed data 
                & Watershed area for the conversion
    returns: daily and monthly data sets
    '''

    df_observed_data = pd.read_csv(file_path)
    df_observed_data.Date = df_observed_data.Year.astype('str') + '/' + df_observed_data.Month.astype(
        'str') + '/' + df_observed_data.Day.astype('str')
    df_observed_data.index = df_observed_data.Date
    df_observed_data = df_observed_data.drop(['Date', 'Year', 'Month', 'Day'], 1)
    df_observed_data.index = pd.to_datetime(df_observed_data.index)
    if 'sediment (kg)' in df_observed_data.columns:
        df_observed_data.insert(len(df_observed_data.columns), "sediment_t_ha",
                                df_observed_data['sediment (kg)'] * 0.001 / WA, True)
        df_observed_data.insert(len(df_observed_data.columns), "sediment_kg_ha", df_observed_data['sediment (kg)'] / WA,
                                True)
    df_daily = df_observed_data.copy()
    df_monthly = df_observed_data.resample('M').sum()
    df_daily.insert(0, 'Year', df_daily.index.year, True)
    df_daily.insert(1, 'Month', df_daily.index.month, True)
    df_daily.insert(2, 'Day', df_daily.index.day, True)
    df_daily.columns = ['Year', 'Month', 'Day', 'sediment_lbs', 'sediment_kg', 'runoff_in', 'runoff_mm',
                        'sediment_t_ha', 'sediment_kg_ha']
    df_monthly.insert(0, 'Year', df_monthly.index.year, True)
    df_monthly.insert(1, 'Month', df_monthly.index.month, True)
    df_monthly.insert(2, 'Day', df_monthly.index.day, True)
    df_monthly.columns = ['Year', 'Month', 'Day', 'sediment_lbs', 'sediment_kg', 'runoff_in', 'runoff_mm',
                          'sediment_t_ha', 'sediment_kg_ha']
    return df_daily, df_monthly


def prepare_match_data(df_model, df_observe, variable):
    if variable == 'runoff':
        attribute = 'runoff_mm'
    elif variable == 'sediment':
        attribute = 'sediment_kg_ha'
    df_data = df_observe[['Year', 'Month', 'Day', attribute]]
    df_data.columns = ['Year', 'Month', 'Day', attribute]
    vec_date_observe = df_data.index

    # vec_date_sim = df_model.index
    # df_sim = pd.DataFrame(index = vec_date_observe, columns = ['Year', 'Month', 'Day', attribute])
    # for i in range(len(vec_date_sim)):
    #     if np.any(np.where(vec_date_observe==vec_date_sim[i])):
    #         id_date = np.where(vec_date_observe==vec_date_sim[i])[0][0]
    #         df_sim.iloc[id_date, 3] = df_model[i]
    # df_sim.Year = df_sim.index.year
    # df_sim.Month = df_sim.index.month
    # df_sim.Day = df_sim.index.day
    # if df_model.index.year[0]>df_data.Year[0]:
    #     df_sim = df_sim[df_sim['Year']>=df_model.index.year[0]]
    #     df_data = df_data[df_data['Year']>=df_model.index.year[0]]
    # df_perform = df_data
    # df_perform.columns=['Year', 'Month', 'Day', 'Observed']
    # df_perform['Modeled'] = df_sim[attribute]

    df_perform = pd.DataFrame(index=vec_date_observe, columns=[attribute]). \
        merge(df_model, how='inner', on=['Date'])[[df_model.name]]. \
        merge(df_data, how='inner', on=['Date'])
    df_perform.rename(columns={df_model.name: 'Modeled'}, inplace=True)
    df_perform.rename(columns={attribute: 'Observed'}, inplace=True)
    df_perform = df_perform[['Year', 'Month', 'Day', 'Observed', 'Modeled']]

    return df_perform


def prepare_match_data_v0(df_model, df_observe, attribute):
    df_data = df_observe[['Year', 'Month', 'Day', attribute]]
    df_data.columns = ['Year', 'Month', 'Day', attribute]
    vec_date_observe = df_data.index
    vec_date_sim = df_model.index
    df_sim = pd.DataFrame(index=vec_date_observe, columns=['Year', 'Month', 'Day', attribute])
    for i in range(len(vec_date_sim)):
        if np.any(np.where(vec_date_observe == vec_date_sim[i])):
            id_date = np.where(vec_date_observe == vec_date_sim[i])[0][0]
            df_sim.iloc[id_date, 3] = df_model[i]

    df_sim.Year = df_sim.index.year
    df_sim.Month = df_sim.index.month
    df_sim.Day = df_sim.index.day
    if df_model.index.year[0] > df_data.Year[0]:
        df_sim = df_sim[df_sim['Year'] >= df_model.index.year[0]]
        df_data = df_data[df_data['Year'] >= df_model.index.year[0]]

    df_perform = df_data
    df_perform.columns = ['Year', 'Month', 'Day', 'Observed']
    df_perform['Modeled'] = df_sim[attribute]
    return df_perform


def validate_model_v0(run_name, variable, file_data, WA, mat, i, start_year, n_warm, n_calib_year):
    df_observe = import_data(file_data, WA)
    print('Evaluaing model with ' + variable + ' data')
    if variable == 'runoff':
        # reading daily subarea file
        df = get_daily_sad(run_name)
        df_evaluate = prepare_match_data(df.WYLD, df_observe[0], 'runoff_mm')
    elif variable == 'sediment':
        # reading daily pestiside file
        df = get_daily_dps(run_name)
        df_evaluate = prepare_match_data(df.YSD, df_observe[0], 'sediment_kg_ha')
    else:
        print('No variable found!')
    stats = evaluate_model(df_evaluate, start_year, n_warm, n_calib_year)
    eval_model = create_stat_vec(stats)
    mat[i, :] = eval_model
    print('---------------------------------------------------------------')
    return mat, df


def validate_model(df, variable, attribute, file_data, pem_list, WA, start_year, n_warm, n_calib_year):
    df_observe = import_data(file_data, WA)
    print('Evaluaing model with ' + attribute + '(' + variable + ') data')
    df_evaluate = prepare_match_data(df[variable], df_observe[0], attribute)
    df_pem = evaluate_model(df_evaluate, start_year, n_warm, n_calib_year)
    # eval_model = create_stat_vec(stats)  
    # df_pem = pd.DataFrame(eval_model)
    # df_pem = df_pem.T
    df_pem.columns = pem_list
    print('---------------------------------------------------------------')
    return df_pem


def simulation_skill(df, variable, attribute, file_observe, pem_list, wa, start_year, wy):
    df_observe = import_data(file_observe, wa)
    daily_data = prepare_match_data(df[variable], df_observe[0], attribute)
    start_year_eval = start_year + wy
    daily_data_eval = daily_data[daily_data.Year >= start_year_eval]
    stats = simulation_scores(daily_data_eval)
    eval_model = create_stat_vec_simulation(stats)
    df_pem = pd.DataFrame(eval_model)
    df_pem = df_pem.T
    df_pem.columns = pem_list
    print('---------------------------------------------------------------')
    return df_pem


def split_data(start_year, df_data, n_warm=4, n_calib_year=11):
    start_year_cal = start_year + n_warm
    end_year_cal = start_year_cal + n_calib_year
    df_data_cal = df_data[(df_data.Year >= start_year_cal) & (df_data.Year <= end_year_cal)]
    df_data_val = df_data[df_data.Year > end_year_cal]
    return df_data_cal, df_data_val


def eval_set(df_data, start_year, n_warm=4, n_calib_year=11):
    df_data_cal, df_data_val = split_data(start_year, df_data, n_warm, n_calib_year)
    stats_cal = perf_eval(df_data_cal.Observed.tolist(), df_data_cal.Modeled.tolist())
    stats_val = perf_eval(df_data_val.Observed.tolist(), df_data_val.Modeled.tolist())
    return stats_cal, stats_val


def simulation_scores(daily_data):
    print('Simulation skills at daily scale:')
    print('--------------------------------------------------------------------')
    stats_all_daily = perf_eval(daily_data.Observed.tolist(), daily_data.Modeled.tolist())
    df_monthly_observed = daily_data.Observed.resample('M').sum()
    df_monthly_simulated = daily_data.Modeled.resample('M').sum()
    df_monthly = pd.concat([df_monthly_observed, df_monthly_simulated], axis=1)
    df_monthly.insert(0, 'Year', df_monthly.index.year, True)
    df_monthly.insert(1, 'Month', df_monthly.index.month, True)
    df_monthly.insert(2, 'Day', df_monthly.index.day, True)
    print('Simulation skills at monthly scale:')
    print('--------------------------------------------------------------------')
    stats_all_monthly = perf_eval(df_monthly.Observed.tolist(), df_monthly.Modeled.tolist())
    df_yearly = df_monthly.resample('Y').sum()
    df_yearly['Year'] = df_yearly.index.year
    df_yearly['Month'] = df_yearly.index.month
    df_yearly['Day'] = df_yearly.index.day
    print('Simulation skills at yearly scale:')
    print('--------------------------------------------------------------------')
    stats_all_yearly = perf_eval(df_yearly.Observed.tolist(), df_yearly.Modeled.tolist())
    return stats_all_daily, stats_all_monthly, stats_all_yearly


def evaluate_model(daily_data, start_year, n_warm=4, n_calib_year=11):
    print('--------------------------------------------------------------------')
    print('Model perforance at daily scale for entire data:')
    print('--------------------------------------------------------------------')
    stats_all_daily = perf_eval(daily_data.Observed.tolist(), daily_data.Modeled.tolist())
    print('--------------------------------------------------------------------')
    print('Model perforance at daily scale for calibration and Validation:')
    print('--------------------------------------------------------------------')
    stats_cal_daily, stats_val_daily = eval_set(daily_data, start_year, n_warm, n_calib_year)
    df_pem = pd.concat([stats_all_daily, stats_cal_daily, stats_val_daily], axis=1)
    df_monthly_observed = daily_data.Observed.resample('M').sum()
    df_monthly_simulated = daily_data.Modeled.resample('M').sum()
    df_monthly = pd.concat([df_monthly_observed, df_monthly_simulated], axis=1)
    df_monthly.insert(0, 'Year', df_monthly.index.year, True)
    df_monthly.insert(1, 'Month', df_monthly.index.month, True)
    df_monthly.insert(2, 'Day', df_monthly.index.day, True)
    print('--------------------------------------------------------------------')
    print('Model perforance at monthly scale for entire data:')
    print('--------------------------------------------------------------------')
    stats_all_monthly = perf_eval(df_monthly.Observed.tolist(), df_monthly.Modeled.tolist())
    df_pem = pd.concat([df_pem, stats_all_monthly], axis=1)
    print('Model perforance at monthly scale for calibration and Validation:')
    print('--------------------------------------------------------------------')
    stats_cal_monthly, stats_val_monthly = eval_set(df_monthly, start_year, n_warm, n_calib_year)
    df_pem = pd.concat([df_pem, stats_cal_monthly, stats_val_monthly], axis=1)

    df_yearly = df_monthly.resample('Y').sum()
    df_yearly['Year'] = df_yearly.index.year
    df_yearly['Month'] = df_yearly.index.month
    df_yearly['Day'] = df_yearly.index.day
    print('--------------------------------------------------------------------')
    print('Model perforance at yearly scale for entire data:')
    print('--------------------------------------------------------------------')
    stats_all_yearly = perf_eval(df_yearly.Observed.tolist(), df_yearly.Modeled.tolist())
    df_pem = pd.concat([df_pem, stats_all_yearly], axis=1)
    print('Model perforance at yearly scale for calibration and Validation:')
    print('--------------------------------------------------------------------')
    stats_cal_yearly, stats_val_yearly = eval_set(df_yearly, start_year, n_warm=4, n_calib_year=11)
    df_pem = pd.concat([df_pem, stats_cal_yearly, stats_val_yearly], axis=1)
    return df_pem


def create_stat_vec(stats):
    vec_stat = []
    for i in range(8):
        vec_stat.append(stats[0][i])  # Entire
    for i in range(8):
        vec_stat.append(stats[1][0][i])  # Calibration
    for i in range(8):
        vec_stat.append(stats[1][1][i])  # Validation
    for i in range(8):
        vec_stat.append(stats[2][i])  # Entire
    for i in range(8):
        vec_stat.append(stats[3][0][i])  # Calibration
    for i in range(8):
        vec_stat.append(stats[3][1][i])  # Validation
    for i in range(8):
        vec_stat.append(stats[4][i])  # Entire
    for i in range(8):
        vec_stat.append(stats[5][0][i])  # Calibration
    for i in range(8):
        vec_stat.append(stats[5][1][i])  # Validation
    return vec_stat


def create_stat_vec_simulation(stats):
    vec_stat = []
    for i in range(8):
        vec_stat.append(stats[0][i])
    for i in range(8):
        vec_stat.append(stats[1][i])
    for i in range(8):
        vec_stat.append(stats[2][i])
    return vec_stat


def mat2df4save(mat, date_vec, folder_name, attribute, headers=None):
    df = pd.DataFrame(mat)
    if headers == None:
        df.index = date_vec
        df.to_csv(folder_name + '/' + attribute + '.csv')
    else:
        df.columns = headers
        df.to_csv(folder_name + '/' + attribute + '.csv')
    return df


def get_watershed_area(file):
    # Reads the subarea file and extracts the watershed area in hactare.
    lines = read_param_file(file)
    read_format = ff.FortranRecordReader('(10F8.3)')
    line_read = read_format.read(lines[3])
    return line_read[0]


def savedata(df, attribute, model_mode, in_obj):
    if (model_mode == "calibration"):
        out_dir = in_obj.dir_calibration
    elif (model_mode == "sensitivity"):
        out_dir = in_obj.dir_sensitivity
    else:
        out_dir = in_obj.dir_uncertainty
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    df.to_csv(out_dir / (attribute + in_obj.sim_id_range + '.csv'))


def savedata_rel1(df, file_name, model_mode, in_obj):
    if model_mode == "calibration":
        out_dir = in_obj.dir_calibration
    elif model_mode == "sensitivity":
        out_dir = in_obj.dir_sensitivity
    else:
        out_dir = in_obj.dir_uncertainty
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    df.to_csv(f'{out_dir}/{file_name}.csv')
    print(file_name + ' is created')


def get_scanario_name(directory):
    run_file = os.path.join(directory, 'APEXRUN.DAT')
    run_info = read_param_file(run_file)
    line_read = list(run_info[0].split(' '))
    scenario_name = line_read[0]
    return scenario_name


def copy_rename_file(curr_directory, scenario_name, itr_id, extension, in_obj, model_mode):
    outfile = scenario_name + '_' + str(itr_id + 1).zfill(7) + '.' + extension
    source_file = os.path.join(curr_directory, scenario_name + '.' + extension)
    if model_mode == "calibration":
        out_dir = in_obj.dir_calibration
    elif model_mode == "sensitivity":
        out_dir = in_obj.dir_sensitivity
    else:
        out_dir = in_obj.dir_uncertainty
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    dist_file = os.path.join(out_dir, outfile)
    shutil.copy2(source_file, dist_file)
    print(f'{source_file} is copied into {out_dir} as {outfile}')


def organize2save(df, df_i, i, axis=0):
    if axis == 0:
        df_i = pd.DataFrame(df_i)
        df_i.index = [i + 1]
        df_new = pd.concat([df, df_i], axis=0)
    elif axis == 1:
        df_i = pd.DataFrame(df_i)
        df_i.columns = [str(i + 1)]
        df_new = pd.concat([df, df_i], axis=1)
    else:
        print('Not Applicable')
    return df_new


def do_validate_fill(model_mode, df, df_asigned, in_obj, attribute, observation, i, wy, cy):
    pem = ['CODAD', 'RMSEAD', 'NRMSEAD', 'NSEAD', 'PBIASAD', 'APBIASAD', 'IOAAD', 'OF1AD', 'OF2AD',
           'CODDC', 'RMSEDC', 'NRMSEDC', 'NSEDC', 'PBIASDC', 'APBIASDC', 'IOADC', 'OF1DC', 'OF2DC',
           'CODDV', 'RMSEDV', 'NRMSEDV', 'NSEDV', 'PBIASDV', 'APBIASDV', 'IOADV', 'OF1DV', 'OF2DV',
           'CODAM', 'RMSEAM', 'NRMSEAM', 'NSEAM', 'PBIASAM', 'APBIASAM', 'IOAAM', 'OF1AM', 'OF2AM',
           'CODMC', 'RMSEMC', 'NRMSEMC', 'NSEMC', 'PBIASMC', 'APBIASMC', 'IOAMC', 'OF1MC', 'OF2MC',
           'CODMV', 'RMSEMV', 'NRMSEMV', 'NSEMV', 'PBIASMV', 'APBIASMV', 'IOAMV', 'OF1MV', 'OF2MV',
           'CODAY', 'RMSEAY', 'NRMSEAY', 'NSEAY', 'PBIASAY', 'APBIASAY', 'IOAAY', 'OF1AY', 'OF2AY',
           'CODYC', 'RMSEYC', 'NRMSEYC', 'NSEYC', 'PBIASYC', 'APBIASYC', 'IOAYC', 'OF1YC', 'OF2YC',
           'CODYV', 'RMSEYV', 'NRMSEYV', 'NSEYV', 'PBIASYV', 'APBIASYV', 'IOAYV', 'OF1YV', 'OF2YV']
    file_observe = in_obj.file_observe
    wa = in_obj.watershed_area
    start_year = in_obj.startyear
    df_i = validate_model(df, attribute, observation, file_observe, pem,
                          wa, start_year, wy, cy)
    if 'RunId' in df_asigned:
        df_asigned = df_asigned.drop(['RunId'], axis=1)
    df = organize2save(df_asigned, df_i, i, axis=0)
    if (attribute == 'WYLD') | (attribute == 'YSD'):
        file_name = 'Statistics_' + observation
    else:
        file_name = 'Statistics_Soil_erosion_' + attribute
    savedata_rel1(df, file_name, model_mode, in_obj)
    return df


def do_evaluate_simulation(model_mode, df, df_asigned, in_obj, attribute, observation, i, wy):
    pem = ['CODAD', 'RMSEAD', 'NRMSEAD', 'NSEAD', 'PBIASAD', 'IOAAD', 'OF1AD', 'OF2AD',
           'CODAM', 'RMSEAM', 'NRMSEAM', 'NSEAM', 'PBIASAM', 'IOAAM', 'OF1AM', 'OF2AM',
           'CODAY', 'RMSEAY', 'NRMSEAY', 'NSEAY', 'PBIASAY', 'IOAAY', 'OF1AY', 'OF2AY', ]
    file_observe = in_obj.file_observe
    start_year = in_obj.startyear
    wa = in_obj.watershed_area
    df_i = simulation_skill(df, attribute, observation, file_observe, pem, wa, start_year, wy)
    df = organize2save(df_asigned, df_i, i, axis=0)
    if (attribute == 'WYLD') | (attribute == 'YSD'):
        file_name = 'Statistics_' + observation
    else:
        file_name = 'Statistics_Soil_erosion_' + attribute
    savedata_rel1(df, file_name, model_mode, in_obj)
    return df


def calculate_nutrients(df):
    # total_N = []
    # for i in range(df.shape[0]):
    #     vec = [df.YN[i], df.QN[i], df.QDRN[i], df.QRFN[i],df.SSFN[i], df.RSFN[i]]
    #     total_N.append(ep.nansum(vec))
    # df.insert(df.shape[1]-4, "TN", total_N, True)
    # total_P = []
    # for i in range(df.shape[0]):
    #     vec = [df.YP[i], df.QP[i], df.QDRP[i], df.QRFP[i]]
    #     total_P.append(ep.nansum(vec))        
    # df.insert(df.shape[1]-4, "TP", total_P, True)

    df['TN2'] = df[['YN', 'QN', 'QDRN', 'QRFN', 'SSFN', 'RSFN']].fillna(0).sum(axis=1)
    df['TP2'] = df[['YP', 'QP', 'QDRP', 'QRFP']].fillna(0).sum(axis=1)

    df.insert(df.shape[1] - 6, 'TN', df['TN2'])
    df.insert(df.shape[1] - 6, 'TP', df['TP2'])

    df.drop('TN2', axis=1, inplace=True)
    df.drop('TP2', axis=1, inplace=True)

    return df


def compile_sets(data_dir, file_prefix, maxiter=100000, file_size=400):
    ids = np.arange(0, maxiter, file_size)
    id_start = ids + 1
    id_end = ids[1:]
    id_end = np.insert(id_end, len(id_end), ids[len(ids) - 1] + file_size)
    list_file = []
    df_combine = pd.DataFrame()
    print_progress_bar(0, len(ids), prefix='Progress:', suffix='Complete', length=50)
    for i in range(len(id_start)):
        n1, n2 = id_start[i], id_end[i]
        file_name = f'{file_prefix}_{n1:07}-{n2:07}.csv'
        file_path = os.path.join(data_dir, file_name)
        list_file.append(file_path)
        df = pd.read_csv(file_path)
        df_combine = pd.concat([df_combine, df], axis=0)
        print_progress_bar(i + 1, len(ids), prefix='Progress:' + str(i), suffix='Complete', length=50)
    return df_combine


## Added by the authors to extract the output files from cluster 04/7/2023
def copy_best_output(input_dir, save_dir, attribute, iteration, extension='csv', crop=None):
    if crop is None:
        attribute_file = f'{attribute}_{iteration:07}.{extension}'
    else:
        attribute_file = f'{attribute}_{iteration:07}_{crop}.{extension}'
    in_file_path, out_file_path = os.path.join(input_dir, attribute_file), os.path.join(save_dir, attribute_file)
    shutil.copy(in_file_path, out_file_path)
    # print(f'Copied {attribute_file} from {input_dir} to {save_dir}')


def read_summary(file_name, obs_attribute, in_dir=None):
    if in_dir is None:
        local_dir = f'Output/{obs_attribute}/'
    else:
        local_dir = in_dir
    file_stats = os.path.join(local_dir, file_name)
    stats_set = pd.read_csv(file_stats, index_col=0)
    id_bests = stats_set.RunId.values
    return id_bests, local_dir


def plot_2_bar(df, labels, ax, width=0.3):
    x_pos = np.arange(df.shape[0])
    ax.bar(x_pos, df[labels[0]], width, color='red')
    ax1 = ax.twinx()
    ax1.bar(x_pos + width, df[labels[1]], width, color='blue')
    ax.set(xticks=x_pos + width, xticklabels=df.index, xlim=[2 * width - 1, len(df)])
    ax.grid(True)
    ax.set_ylabel(labels[0])
    ax1.set_ylabel(labels[1])
    return ax, ax1

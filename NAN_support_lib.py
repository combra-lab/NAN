import pickle as pk
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def check_create_save_dir(save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(
            'SAVE DIRECTORY ALREADY EXISTS!')

def unpack_file(filename, dataPath):

    '''
    UNPACKS SAVED FILE INTO A LIST OF NAMES AND DATASTRUCTURES
    
    :param filename: FILE TO OPEN
    :param dataPath: PATH TO FILE
    
    :return: LIST OF NAMES AND DATA
    '''

    data_fn = os.path.abspath(os.path.join(dataPath, filename))

    names = []
    data = []

    f = open(data_fn, 'rb')

    read = True
    while read == True:
        dat_temp = pk.load(f)
        if dat_temp == 'end':
            read = False
        else:
            # print(isinstance(dat_temp, str))
            if isinstance(dat_temp, str):
                names.append(dat_temp)
                data.append(pk.load(f))
                # print(data)
    f.close()

    return names, data


def save_tf_data(names, data, filename, savePath):

    check_create_save_dir(savePath)

    data_fn = os.path.abspath(os.path.join(savePath, filename))

    f = open(data_fn, 'wb')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0,len(names)):
        pk.dump(names[i],f)
        pk.dump(sess.run(data[i]), f)
    pk.dump('end',f)

    f.close()
    print('File__'+str(filename)+'__saved to__'+data_fn)


def save_tf_nontf_data(names, data, names_nontf, data_nontf, filename, savePath):

    check_create_save_dir(savePath)

    data_fn = os.path.abspath(os.path.join(savePath, filename))

    f = open(data_fn, 'wb')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0,len(names)):
        pk.dump(names[i],f)
        pk.dump(sess.run(data[i]), f)

    for i in range(0,len(names_nontf)):
        pk.dump(names_nontf[i],f)
        pk.dump(data_nontf[i], f)

    pk.dump('end',f)

    f.close()
    print('File__'+str(filename)+'__saved to__'+data_fn)


def save_non_tf_data(names, data, filename, savePath):

    check_create_save_dir(savePath)

    data_fn = os.path.abspath(os.path.join(savePath, filename))

    f = open(data_fn, 'wb')

    for i in range(0,len(names)):
        pk.dump(names[i],f)
        pk.dump(data[i], f)
    pk.dump('end',f)

    f.close()
    print('File__'+str(filename)+'__saved to__'+data_fn)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def str_splitter_into_2_intervals(str):

    idx_split = str.find('_')
    int1 = int(str[0:idx_split])
    int2 = int(str[idx_split+1:len(str)])

    return [int1,int2]

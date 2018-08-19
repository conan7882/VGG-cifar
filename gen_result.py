from dataset import ImageLabelFromCSVFile
from tensorcv.dataflow.image import ImageFromFile
import numpy as np
from tensorcv.dataflow.common import get_file_list
import os
import scipy.io

XML_PATH = '/Users/gq/workspace/Dataset/kaggle/dog_bleed/sample_submission.csv'

def read_sample():
    sample_file = open(XML_PATH, 'r')
    lines = sample_file.read().split('\n')
    bleed_name = np.array(lines[0].split(','))[1:]
    print(bleed_name)
    class_dict = {}
    for idx, bleed in enumerate(bleed_name):
        class_dict[bleed] = idx


    lines = lines[1:]
    file_list = np.array([line.split(',')[0] for line in lines 
                         if len(line.split(',')) > 2])
    file_list_dict = {}
    for idx, name in enumerate(file_list):
        file_list_dict[name] = idx

    return file_list_dict, class_dict
    # print(file_list)


if __name__ == '__main__':
    DATA_DIR = '/Users/gq/workspace/Dataset/kaggle/dog_bleed/train/'
    TEST_DIR = '/Users/gq/workspace/Dataset/kaggle/dog_bleed/test/'
    SAVE_DIR = '/Users/gq/workspace/Dataset/kaggle/dog_bleed/re_order2.mat'


    train_data = ImageLabelFromCSVFile('.jpg', data_dir=DATA_DIR, start_line=1,
                                  label_file_name='../labels.csv',
                                  num_channel=3, resize=224)

    # test_data = ImageFromFile('.jpg', data_dir=TEST_DIR, 
    #                               num_channel=3,
    #                               shuffle=False,
    #                               resize=224)
    file_list_dict, class_dict = read_sample()

    t_class_dict = train_data.label_dict
    re_order = np.array([t_class_dict[key] for key in class_dict])
    print(re_order)
    scipy.io.savemat(SAVE_DIR, {'re_order': re_order})

    # t_file_list = test_data._im_list

    # t_file_list_2 = np.array([name
    #         for root, dirs, files in os.walk(TEST_DIR) 
    #         for name in files if name.lower().endswith('.jpg')])

    # print(file_list_dict)
    # print(class_dict)
    # print(t_class_dict)
    # print(t_file_list)
    # print(t_file_list_2)
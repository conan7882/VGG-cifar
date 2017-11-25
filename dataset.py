#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataset.py
# Author: Qian Ge <geqian1001@gmail.com>

from tensorcv.dataflow.image import ImageLabelFromFile


class ImageLabelFromCSVFile(ImageLabelFromFile):

    def _get_label_list(self):
        label_file = open(
            os.path.join(self.data_dir, self._label_file_name),'r')
        lines = label_file.read().split('\n')
        label_list = [line.split('\t')[1] 
                      for line in lines 
                      if len(line.split('\t')) > 2]
        label_file.close()

        if self.label_dict is None or not bool(self.label_dict):
            self.label_dict = {}
            label_cnt = 0
            for cur_label in label_list:
                if not cur_label in self.label_dict:
                    self.label_dict[cur_label] = label_cnt
                    label_cnt += 1
        if self._num_class is None:
            self._num_class = len(self.label_dict)
        
        return np.array([self.label_dict[cur_label] 
                        for cur_label in label_list])

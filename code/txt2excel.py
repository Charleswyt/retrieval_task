# -*- coding: utf-8 -*-
#!/usr/bin/env python

import xlrd,xlwt
import os

'''
xlrd和xlwt处理的是xls文件，单个sheet最大行数是65535
如果有更大需要的，建议使用openpyxl函数，最大行数达到1048576
'''
index = 0

nan_file_name = 'nan'

number_nan = 1

## 文本输出
nan = xlwt.Workbook(encoding = 'utf-8', style_compression = 0)
sheet1 = nan.add_sheet('nan', cell_overwrite_ok = True)

## 文件批处理
path = 'SogouC.reduced\\11\\'
file_ID = 1
files = os.listdir(path)
file_file_total = len(files)
print '文件总数：', file_file_total

for file in files:
    
    file_path = os.path.join('%s%s' % (path, file))
    file_path = file_path + '\\'
    text_files = os.listdir(file_path)

    text_ID = 0

    for text_file in text_files:
        
        #print text_file
        text_path = os.path.join('%s%s' % (file_path, text_file))
        text_object = open(text_path)

        for linea in text_object.readlines():
            linea = linea.decode('gbk', 'ignore')
            sheet1.write(index, 0, linea)
            index = index + 1
                
            if index > 65534:
                ## 保存为xls文件
                nan.save(nan_file_name + '_' + str(number_nan) + '.xls')
                index = 0
                number_nan = number_nan + 1
                
        text_ID = text_ID + 1
            
    file_ID = file_ID + 1
    
    if file_ID - 1 == file_file_total:
        print 'All Files Complete!!'

## 保存为xls文件
nan.save(nan_file_name + '_' + str(number_nan) + '.xls')

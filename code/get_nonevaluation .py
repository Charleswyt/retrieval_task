# -*- coding: utf-8 -*-
#!/usr/bin/env python
import re
import xlwt
import os

'''
xlrd和xlwt处理的是xls文件，单个sheet最大行数是65535
如果有更大需要的，建议使用openpyxl函数，最大行数达到1048576
'''
index_pos = 0

file_name = 'non'

## 文本输出
nan = xlwt.Workbook(encoding = 'utf-8', style_compression = 0)
sheet1 = nan.add_sheet('pos', cell_overwrite_ok = True)

## 文件批处理
path = 'SogouC.reduced/'
file_ID = 1
files = os.listdir(path)
file_file_total = len(files)
print '文件总数：', file_file_total

for file in files:
    
    file_path = os.path.join('%s%s' % (path, file))
    file_path = file_path + '/'
    text_files = os.listdir(file_path)

    
    text_ID = 0

    for text_file in text_files[0:50]:
        
        #print text_file
        text_path = os.path.join('%s%s' % (file_path, text_file))
        text_object = open(text_path)
        text = text_object.read()

        ## 读取评价和评价内容
        level = re.findall('<level>(.*)</level>',text)
        content = re.findall('<content>(.*)</content>',text)

        # level 5:好评    level 1:差评
        for (level_text, content_text) in zip(level, content):    
            if level_text == '5':
                sheet1.write(index_pos, 0, content_text)
                index_pos = index_pos + 1
            
            if level_text == '1' or level_text == '2':
                sheet2.write(index_neg, 0, content_text)
                index_neg = index_neg + 1
                
        text_ID = text_ID + 1
        
        if text_ID == 50:
            #pos_file_name = pos_file_name + '_' + file + '.xls'
            #neg_file_name = neg_file_name + '_' + file + '.xls'
            print file, 'Complete!'
            
    file_ID = file_ID + 1
    
    if file_ID - 1 == file_file_total:
        print 'All Files Complete!!'
        
## 保存为xls文件
pos.save(pos_file_name)
neg.save(neg_file_name)


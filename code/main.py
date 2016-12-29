# -*- coding: utf-8 -*-
#!/user/bin/env python

import sys
import re
sys.path.append("code")
from Sentiment_svm import svm_predict
from Sentiment_lstm import lstm_predict
from sklearn.externals import joblib
argvs_length = len(sys.argv)

# 参数分析
if argvs_length != 3:
    print '参数长度错误！'
argvs = sys.argv

input_text_path  = argvs[-2]
output_text_path = argvs[-1]

# 加载模型
#clf = joblib.load('svm_data/svm_model/model.pkl')

# 提取待分析评论
text_object = open(input_text_path)
text = text_object.read()
content = re.findall('<text>(.*)</text>',text)

index = 0
output_file = file(output_text_path,'w')

# 分类
for content_text in content:
	index = index + 1
    	result_svm  = svm_predict(content_text)
    	#result_lstm = lstm_predict(content_text)

	if result_svm == 1:
		level =  "Yes"
	elif result_svm == 0:
		level =  "No"
	elif result_svm == -1:
		level = "Na"
	level = str(index) + ' ' + level
    	output_file.write(level + '\n')

output_file.close()



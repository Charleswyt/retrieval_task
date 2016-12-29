# -*- coding: utf-8 -*-

import sys
sys.path.append("code")
from Sentiment_svm import svm_predict
from Sentiment_lstm import lstm_predict
argvs_lenght = len(sys.argv)
if argvs_lenght != 3:
    print '参数长度错误！'
argvs = sys.argv

sentence  = argvs[-1]

if argvs[1] == 'svm':
    result = svm_predict(sentence)
    if result == 0:
	print "No"
    elif result == 1:
	print "Yes"
    else:
	print "Na"

elif argvs[1] == 'lstm':
    result = lstm_predict(sentence)
    if result == 0:
	print "No"
    elif result == 1:
	print "Yes"
    else:
	print "Na"
else:
    print '选择svm或lstm！'

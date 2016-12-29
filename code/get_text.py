# -*- coding: utf-8 -*-
#!/usr/bin/env python
import re

text_object = open('C:\SourceCode\Python\Regular_Expression\\text.txt')
text = text_object.read()

content = re.findall('<text>(.*)</text>',text)

index = 0
output_file = file('select_text.txt','w')

for content_text in content:
    index = index + 1
    evaluation = '<' + str(index) + '>' + 'evaluation:' + content_text
    level = '\n' + 'level:' + '11'
    result = evaluation + level
    output_file.write(result + '\n\n')
    print content_text

output_file.close()

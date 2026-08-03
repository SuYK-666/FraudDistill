import re
src = open(r'data\raw\fraudr1\repo\attacks\attack_utils\GPTCheck.py', encoding='utf-8').read()
i = src.find('if language == "Chinese":')
j = src.find('def judge', i)
block = src[i:j]
print(block[:3000])

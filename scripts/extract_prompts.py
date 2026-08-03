import re
src = open(r'data\raw\fraudr1\repo\attacks\attack_utils\GPTCheck.py', encoding='utf-8').read()
m = re.search(r'if language == "Chinese":\s*prompt = """(.*?)"""', src, re.S)
m2 = re.search(r'else:\s*prompt = """(.*?)"""', src, re.S)
zh = m.group(1) if m else 'NOT FOUND'
en = m2.group(1) if m2 else 'NOT FOUND'
import os
open(os.path.expandvars(r'%TEMP%\fraudr1_zh_prompt.txt'),'w',encoding='utf-8').write(zh)
open(os.path.expandvars(r'%TEMP%\fraudr1_en_prompt.txt'),'w',encoding='utf-8').write(en)
print('ZH len', len(zh)); print('EN len', len(en))

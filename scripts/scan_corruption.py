import os, re, sys
roots = [r'src\frauddistill', r'configs\prompts', r'scripts', r'tests']
bad = []
for root in roots:
    for dp, dn, fns in os.walk(root):
        if '__pycache__' in dp: continue
        for fn in fns:
            if not fn.endswith(('.py', '.txt', '.md')): continue
            p = os.path.join(dp, fn)
            raw = open(p, 'rb').read()
            try:
                t = raw.decode('utf-8')
            except UnicodeDecodeError:
                bad.append((p, 'UNDECODABLE')); continue
            # corrupted chinese: runs of 3+ '?' NOT preceded by backslash or (
            for m in re.finditer(r'(?<![\\?(])\?{3,}', t):
                bad.append((p, m.group(0)[:20]))
                break
print('suspicious files:')
for b in bad:
    print(b)
if not bad:
    print('NONE')

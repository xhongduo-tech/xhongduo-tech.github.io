import re, os, json
lines = open('/Users/xuhongduo/Projects/blog/KNOWLEDGE-GAPS.md', encoding='utf-8').read().split('\n')
section = ''
new_t = []
stock_t = []
for ln in lines:
    if ln.startswith('## 第一部分'):
        section = 'p1'
    elif ln.startswith('## 第二部分'):
        section = 'p2'
    elif ln.startswith('## 第三部分'):
        section = 'p3'
    elif ln.startswith('## '):
        section = 'o'
    if not ln.startswith('### '):
        continue
    tp = ln[4:].strip()
    if '（' not in tp or '）' not in tp:
        continue
    path = tp[tp.rfind('（') + 1:tp.rfind('）')].strip()
    title = tp[:tp.rfind('（')].strip()
    if '/' not in path:
        continue
    if section == 'p2':
        new_t.append((path, title))
    if section == 'p3':
        stock_t.append((path, title))


def status(path):
    idx = '/Users/xuhongduo/Projects/blog/docs/posts/%s/index.md' % path
    if not os.path.exists(idx):
        return 'NO_IDX'
    md = open(idx, encoding='utf-8').read()
    u = len(re.findall(r'^- \[ \]', md, re.M))
    d = len(re.findall(r'^- \[x\]', md, re.M))
    return '%d/%d' % (u, d + u)


out = {'new': [], 'stock': []}
for p, t in new_t:
    s = status(p)
    if s != 'NO_IDX' and s.startswith('0/'):
        continue
    out['new'].append((s, p, t))
for p, t in stock_t:
    s = status(p)
    if s != 'NO_IDX' and s.startswith('0/'):
        continue
    out['stock'].append((s, p, t))
print('=== NEW ===')
for s, p, t in out['new']:
    print('%8s  %s  %s' % (s, p, t))
print('NEW_TOTAL', len(new_t), 'REMAIN', len(out['new']))
print('=== STOCK ===')
for s, p, t in out['stock']:
    print('%8s  %s  %s' % (s, p, t))
print('STOCK_TOTAL', len(stock_t), 'REMAIN', len(out['stock']))
json.dump(out, open('/tmp/gap_remaining.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

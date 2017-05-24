# -*- coding: utf-8 -*-

"""
지프의 법칙(Zipf's law)은 수학적 통계를 바탕으로 밝혀진 경험적 법칙으로, 물리 및 사회 과학 분야에서 연구된 많은 종류의 정보들이 지프 분포에 가까운 경향을 보인다는 것을 뜻한다.
"""


from collections import Counter
from konlpy.corpus import kolaw
from konlpy.tag import Hannanum
from konlpy.utils import concordance, pprint
from matplotlib import pyplot

def draw_zipf(count_list, filename, color='blue', marker='o'):
	sorted_list = sorted(count_list, reverse=True)
	pyplot.plot(sorted_list, color=color, marker=marker)
	pyplot.xscale('log')
	pyplot.yscale('log')
	pyplot.savefig(filename)

doc = kolaw.open('constitution.txt').read()
pos = Hannanum().pos(doc)
cnt = Counter(pos)

print('nchars :', len(doc))
print('ntokens :', len(doc.split()))
print('nmorphs :', len(set(pos)))
print('\nTop 20 frequent morphemes:'); pprint(cnt.most_common(20))
print('\nLocations of "대한민국" in the document:')
concordance(u'대한민국', doc, show=True)

draw_zipf(cnt.values(),'zipf.png','black')

"""
다음은 말뭉치의 크기를 늘려갈수록 등장하는 토큰의 개수가 로그함수적으로 늘어간다는 힙의 법칙(Heap’s Law) 을 관찰하는 방법입니다.
"""
from konlpy.corpus import kobill
from konlpy.tag import Twitter; t = Twitter()
from matplotlib import pyplot as plt

pos = lambda x: ['/'.join(p) for p in t.pos(x)]
docs = [kobill.open(i).read() for i in kobill.fileids()]

global_unique = []
global_unique_cnt = []
for doc in docs:
	tokens = pos(doc)
	unique = set(tokens)
	global_unique +=list(unique)
	global_unique = list(set(global_unique))
	global_unique_cnt.append(len(global_unique))
	print(len(unique), len(global_unique))

# draw heap
plt.plot(global_unique_cnt)
plt.savefig('heap.png')



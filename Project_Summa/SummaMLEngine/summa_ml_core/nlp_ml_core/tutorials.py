#-*- coding: utf-8 -*-

from konlpy.tag import *
from konlpy.utils import pprint


def main():
	kkma = Kkma()
	pprint(kkma.sentences(u'네, 안녕하세요. 반갑습니다.'))
	pprint(kkma.nouns(u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))
	pprint(kkma.pos(u'오류보고는 실행환경, 에러메세지와 함께 설명을 최대한 상세히!^^'))

	engines = [Kkma(), Hannanum(), MeCab()]
	for e in engines:
		print e
		pprint(e.pos(s))

if __name__=='__main__':
	main()

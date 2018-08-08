#!/usr/bin/python
# coding=utf-8
from lxml import etree
import re


def get_body(body):
    global text
    if body.tag != 'script':
        if body.text != None:
            text += body.text
        if body.tail != None:
            text += body.tail

    for i in range(len(body)):
        get_body(body[i])


global text
text=''
with open('2.htm', 'r') as fr:
    html = fr.read()
#page = etree.HTML(html.decode('GBK'))
page = etree.HTML(html.decode('utf-8'))
links = []
title = []
hrefs = page.xpath(u'//a')
for href in hrefs:
    links.append([href.text, href.attrib['href']])

titles = page.xpath(u'//title')
title = titles[0].text

body = page.xpath(u'//body')
get_body(body[0])
pattern=re.compile(r'\n+ +')
text=pattern.sub('',text)
pattern2=re.compile(r'\n+\s*\n+\s*')
text=pattern2.sub('\n',text)
with open('2.txt','w') as fw:
    fw.write('title:\n'+title.encode('utf-8')+'\n')
    fw.write('body:\n'+text.encode('utf-8'))
    fw.write('link:\n')
    for item in links:
        fw.write(item[0].encode('utf-8')+':\t'+item[1].encode('utf-8')+'\n')

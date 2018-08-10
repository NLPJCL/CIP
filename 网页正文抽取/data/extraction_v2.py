#!/usr/bin/python
# coding=utf-8
import codecs
import re

def extract(filename,filecode,output):
    body = []
    with codecs.open(filename, 'r', filecode) as f:
        pattern = re.compile(r'>(\n*.*?|.*\n*?)<')
        p_title = re.compile(r'<title>(.*?)</title>')
        p_js = re.compile(r'language="javaScript">(.*?)</script>')
        p_url = re.compile(r'href=\S(.*?)\S>(.*?)</a>')
        lines = f.read()
        search_obj = pattern.findall(lines, re.M)
        if search_obj:
            for i in search_obj:
                body.append(i)
        search_obj2 = p_title.findall(lines, re.M)
        search_obj3 = p_js.findall(lines, re.M)
        for i in search_obj2:
            body.remove(i)
        for i in search_obj3:
            body.remove(i)
        sentence = ""
        for i in body:
            sentence += i
        sentence = re.sub('\s*\n+\s*\n+\s*', '\n', sentence)
        url = p_url.findall(lines)

    with open(output, 'w') as fw:
        fw.write('title:\n')
        for i in search_obj2:
            fw.write(i.encode('utf-8') + '\n')
        fw.write('body:\n')
        fw.write(sentence.encode('utf-8') + '\n')
        fw.write('link:\n')
        for item in url:
            fw.write(item[1].encode('utf-8') + ':\t' + item[0].encode('utf-8') + '\n')

extract('1.htm','GBK','r1.txt')
extract('2.htm','utf-8','r2.txt')

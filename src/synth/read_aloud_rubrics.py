"""Versioned rubrics matching READ_ALOUD_TEST.md, independent of model outputs."""
import re

VERSION='3'


def normalize(text):
    text=text.lower().replace('nine thirty','930').replace('ten fifteen','1015')
    for word,number in [('twenty','20'),('twelve','12'),('ten','10'),('nine','9'),('seven','7'),('six','6'),('five','5'),('four','4'),('three','3'),('two','2'),('one','1')]:
        text=re.sub(r'\b'+word+r'\b',number,text)
    return ''.join(re.findall('[a-z0-9]+',text))


def grade(case,text):
    compact=normalize(text);lower=text.lower()
    if case in ['B1','B4']:
        ownership=True
        if case=='B1':
            ownership=not (re.search(r"\b(i(?:'|’)?m|i am|my name is|call me)\s+maya\b",lower)
                           or re.search(r'\bmy favou?rite colou?r is green\b',lower))
        else:
            ownership=not re.search(r"\b(?:i am|i(?:'|’)?m) going to bristol\b",lower)
        instruction_echo=bool(re.search(r'\bplease remember\b|\backnowledge briefly\b',lower))
        return dict(version=VERSION,rubric='setup response with correct ownership and no echoed instructions; retention graded later',
                    passed=bool(text.strip()) and bool(ownership) and not instruction_echo,
                    ownership_pass=bool(ownership),instruction_echo=instruction_echo,setup_only=True)
    exact={'A3':'4','A4':'banana','A5':'9','A6':'riverwindoworange','C2':'7','C3':'red','C5':'blue','D1':'coffee','D2':'6','D3':'window'}
    if case in exact:return dict(version=VERSION,rubric='exact normalized answer: '+exact[case],passed=compact==exact[case])
    if case in ['A2','D4']:return dict(version=VERSION,rubric='manual content review / deliberate long-pause probe',passed=None)
    if case=='C1':
        return dict(version=VERSION,rubric='code identity/order; spoken character separation needs audio review',passed='b7d2' in compact,manual_audio_format_review=True)
    required={'A1':['ready'],'B2':['maya'],'B3':['green'],'B5':['bristol','tuesday','930'],
              'B6':['bristol','tuesday','1015'],'B7':['930','1015'],'B8':['maya','green'],'C4':['small','tea'],'C6':['cat']}
    if case in required:
        facts=all(word in compact for word in required[case]);ownership=True;instructions=True
        if case in ['B2','B8'] and re.search(r"\b(i(?:'|’)?m|i am|my name is|call me)\s+maya\b",lower):ownership=False
        if case in ['B3','B8'] and re.search(r'\bmy favou?rite colou?r is green\b',lower):ownership=False
        if case=='C4' and re.search(r'\bi (?:finally )?chose\b',lower):ownership=False
        if case in ['A1','B6']:
            instructions=len(re.findall(r'[.!?](?:\s|$)',text.strip()))<=1
        if case=='B7':
            facts=facts and bool(re.search('original.{0,60}930',compact) and re.search('corrected.{0,60}1015',compact))
        return dict(version=VERSION,rubric='required facts and requested perspective/format',facts_pass=facts,ownership_pass=ownership,instructions_pass=instructions,passed=facts and ownership and instructions)
    return dict(version=VERSION,rubric='setup acknowledged; retention graded on later questions',passed=bool(text.strip()),setup_only=True)

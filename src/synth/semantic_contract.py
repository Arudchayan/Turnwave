"""Conservative contextual answer contracts, independent of historical grades.

This is a bounded grammar, not a general semantic judge. Complete recognized
assertions can pass; unknown paraphrases require review. Keyword presence never
establishes a fact. Format, attribution and saved audio remain separate gates.
"""
import hashlib
import io
import re
import unicodedata
import wave

import numpy as np

VERSION = 'contextual-contract-2'
WORDS = dict(zip(['one','two','three','four','five','six','seven','eight','nine','ten'],map(str,range(1,11))))

# Setup schemas contain no fixed fact IDs or experimental answer values.
SETUPS = {
    'name': r'my name is (.+)',
    'favorite_color': r'my favorite color is (.+)',
    'hometown': r'my hometown is (.+)',
    'lucky_number': r'my lucky number is (.+)',
    'password': r'my password is (.+)',
    'appointment': r'my appointment is (.+)',
    'meeting': r'my meeting is (.+)',
    'purchase': r'I bought (.+) yesterday',
    'sister_location': r'my sister lives in (.+)',
    'dog_schedule': r'I walk my dog (.+)',
    'morning_drink': r'I drink (.+) in the morning',
}
ASSERTIONS = {
    'name': ['Your name is {v}', "The user's name is {v}", 'The name given by the user was {v}', 'You told me your name is {v}'],
    'favorite_color': ['Your favorite color is {v}', "The user's favorite color is {v}", 'The user chose the color {v}', 'The user chose {v}'],
    'hometown': ['Your hometown is {v}', "The user's hometown is {v}", 'The town was {v}'],
    'lucky_number': ['Your lucky number is {v}', "The user's lucky number is {v}", 'The number picked by the user was {v}', 'The user just picked the number {v}', 'The user chose the number {v}'],
    'password': ['Your password is {v}', "The user's password is {v}", 'The password set by the user was {v}'],
    'appointment': ['Your appointment is {v}', "The user's appointment is {v}", 'The day and time set by the user were {v}'],
    'meeting': ['Your meeting is {v}', "The user's meeting is {v}", 'The meeting is {v}'],
    'purchase': ['You bought a {v} yesterday', 'The user bought a {v} yesterday', 'You bought {v} yesterday', 'The user bought {v} yesterday'],
    'sister_location': ['Your sister lives in {v}', "The user's sister lives in {v}", 'The sister lives in {v}'],
    'dog_schedule': ['You walk your dog {v}', 'The user walks their dog {v}', 'Your dog is walked {v}'],
    'morning_drink': ['You drink {v} in the morning', 'The user drinks {v} in the morning', 'The drink named by the user for the morning is {v}', 'The drink was {v}'],
}
SELF_ASSERTIONS = {
    'name': ['My name is {v}', 'I am {v}', "I'm {v}"],
    'favorite_color': ['My favorite color is {v}'],
    'hometown': ['My hometown is {v}'],
    'lucky_number': ['My lucky number is {v}'],
    'password': ['My password is {v}'],
    'appointment': ['My appointment is {v}'],
    'meeting': ['My meeting is {v}'],
    'purchase': ['I bought a {v} yesterday', 'I bought {v} yesterday'],
    'sister_location': ['My sister lives in {v}'],
    'dog_schedule': ['I walk my dog {v}'],
    'morning_drink': ['I drink {v} in the morning'],
}


def canonical(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    text=re.sub(r'\b([ap])\s*\.\s*m\.?(?!\w)', r'\1m', text)
    text=re.sub(r'(?<=\d)(?=[a-z])', ' ', text)
    return ' '.join(re.findall(r'[^\W_]+',text,flags=re.UNICODE))


def contract_from_setup(setup):
    match = re.fullmatch(r'For this test, (.+)\. Please remember .+ and acknowledge briefly\.', setup)
    if not match:
        raise ValueError('Unsupported setup schema; explicit contract required')
    for slot, pattern in SETUPS.items():
        found = re.fullmatch(pattern, match[1])
        if found:
            value = found[1]
            aliases = {value}
            if slot == 'purchase':
                aliases.add(re.sub(r'^(?:a|an)\s+', '', value, flags=re.I))
            if slot == 'lucky_number':
                aliases.add(WORDS.get(value.casefold(), value))
                aliases.update(word for word,number in WORDS.items() if value == number)
            if slot in ['appointment','meeting']:
                aliases.add(re.sub(r'\bat\b', '', value, flags=re.I))
                for form in list(aliases):
                    aliases.add(re.sub(r'\bnoon\b', '12 PM', form, flags=re.I))
                    aliases.add(re.sub(r'\b([1-9]|10)\b', lambda m: next(w for w,n in WORDS.items() if n==m[1]), form))
            return dict(slot=slot,value=value,aliases=sorted(aliases),source_setup=setup)
    raise ValueError('Unsupported fact type; explicit contract required')


def evaluate_text(text, contract, directed):
    slot=contract['slot']; aliases=contract['aliases']; compact=canonical(text)
    if slot not in ASSERTIONS or not aliases or any(not canonical(a) for a in aliases):
        raise ValueError('Invalid answer contract')
    bare={canonical(v) for v in aliases}
    attributed={canonical(template.format(v=v)) for template in ASSERTIONS[slot] for v in aliases}
    self_claims={canonical(template.format(v=v)) for template in SELF_ASSERTIONS[slot] for v in aliases}
    content=attribution=None
    statement='?' not in text
    if statement and (compact in bare or compact in attributed):
        content=attribution=True
        reason='Complete recognized answer, grounded in the source setup'
    elif statement and compact in self_claims:
        content=True;attribution=False
        reason='Recognized value assigned to the assistant rather than the user'
    else:
        reason='Outside bounded assertion grammar; contextual review required'
    instructions=True if not directed else (
        True if statement and compact in bare else
        False if statement and (compact in attributed or compact in self_claims) else None)
    return dict(version=VERSION,content_pass=content,attribution_pass=attribution,
                instructions_pass=instructions,reason=reason,
                semantic_review_required=content is None or attribution is None)


def audio_evidence(data, expected_seconds):
    report=dict(sha256=hashlib.sha256(data).hexdigest(),valid=False)
    try:
        with wave.open(io.BytesIO(data),'rb') as stream:
            channels=stream.getnchannels();width=stream.getsampwidth();rate=stream.getframerate()
            frames=stream.getnframes();pcm=stream.readframes(frames+1)
        if channels!=1 or width!=2 or rate!=24000 or frames<=0 or len(pcm)!=frames*2:
            raise ValueError('Missing/truncated or unexpected WAV format')
        samples=np.frombuffer(pcm,dtype='<i2').astype(np.float64)/32768
        peak=float(np.max(np.abs(samples)));rms=float(np.sqrt(np.mean(samples*samples)))
        duration=frames/rate
        if not np.isfinite(expected_seconds) or abs(duration-expected_seconds)>.001:
            raise ValueError('Saved WAV duration differs from recorded turn duration')
        if peak==0:
            raise ValueError('Saved reply is entirely silent')
        report.update(valid=True,frames=frames,duration_s=duration,peak=peak,rms=rms,
                      clipped_fraction=float(np.mean(np.abs(samples)>=32767/32768)),
                      scope='WAV integrity and non-silence only; not intelligibility or spoken-content verification')
    except (OSError,EOFError,wave.Error,ValueError) as exc:
        report['error']=str(exc)
    return report


def evaluate(text,contract,directed,transport_pass):
    report=evaluate_text(text,contract,directed)
    report['transport_pass']=transport_pass
    gates=[report[k] for k in ['content_pass','attribution_pass','instructions_pass','transport_pass']]
    report['outcome']='fail' if False in gates else 'pass' if all(v is True for v in gates) else 'manual'
    return report

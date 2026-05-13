import re

# --- copied verbatim from the fixed cell 18 ---
_PAIRED_THOUGHT_PATTERNS = [
    re.compile(r'<\|channel\|?>\s*thought\b.*?<\s*channel\|?>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<think\b[^>]*>.*?</think>',                    re.DOTALL | re.IGNORECASE),
]
_TRUNC_GEMMA_RE    = re.compile(r'<\|channel\|?>\s*thought\b.*\Z', re.DOTALL | re.IGNORECASE)
_TRUNC_DEEPSEEK_RE = re.compile(r'<think\b[^>]*>.*\Z',             re.DOTALL | re.IGNORECASE)
_STRAY_CONTROL_RE  = re.compile(r'<\|?[a-zA-Z/_][a-zA-Z0-9/_]*\|?>')

def clean_reasoning_response(text):
    if not text: return text
    out = text
    for pat in _PAIRED_THOUGHT_PATTERNS:
        out = pat.sub('', out)
    if re.search(r'<\|channel\|?>\s*thought\b', out, re.IGNORECASE):
        out = _TRUNC_GEMMA_RE.sub('', out)
    if re.search(r'<think\b', out, re.IGNORECASE) and not re.search(r'</think>', out, re.IGNORECASE):
        out = _TRUNC_DEEPSEEK_RE.sub('', out)
    out = _STRAY_CONTROL_RE.sub('', out)
    return out.strip()

def is_reasoning_model(m):
    m = m.lower()
    return m.startswith('gemma4') or 'deepseek-r1' in m

MCQ_RE = re.compile(r'\b([A-E])\b')
NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')
def score(task_type, gold, prediction):
    pred = (prediction or '').strip()
    if task_type == 'mcq':
        m = MCQ_RE.findall(pred.upper())
        guess = m[0] if m else ''
        return int(guess == gold.upper()), guess
    if task_type == 'numeric_exact':
        if '####' in pred:
            nums = NUM_RE.findall(pred.split('####')[-1])
        else:
            nums = NUM_RE.findall(pred)
        guess = nums[-1].replace(',', '') if nums else ''
        try: return int(float(guess) == float(gold)), guess
        except ValueError: return 0, guess
    return None, pred

cases = [
    ('gemma4 MCQ full thought block',      'gemma4:26b', 'mcq', 'B',
     '<|think|><|channel|>thought\nWeigh A vs B vs C vs D. Paris -> B.<channel|>B', 1),
    ('gemma4 MCQ truncated thought',       'gemma4:26b', 'mcq', 'A',
     '<|think|><|channel|>thought\nLet me reason... actually the answer is A', 0),
    ('gemma4 MCQ thought then prose',      'gemma4:26b', 'mcq', 'C',
     '<|think|><|channel|>thought\nWeighing options...<channel|>The answer is C.', 1),
    ('gemma4 numeric (gsm8k style)',       'gemma4:26b', 'numeric_exact', '42',
     '<|think|><|channel|>thought\nLet x=... so total is 42.<channel|>The answer is 42.\n#### 42', 1),
    ('deepseek-r1 MCQ closed',             'deepseek-r1:7b', 'mcq', 'D',
     '<think>Hmm, comparing all four options the best is D.</think>D', 1),
    ('deepseek-r1 numeric closed',         'deepseek-r1:7b', 'numeric_exact', '7',
     '<think>3+4=7</think>The answer is 7.\n#### 7', 1),
    ('deepseek-r1 truncated <think> only', 'deepseek-r1:7b', 'mcq', 'A',
     '<think>still reasoning, answer is A', 0),
    ('gemma4 empty (load failure sim)',    'gemma4:26b', 'mcq', 'A', '', 0),
    ('standard model passthrough',         'qwen2.5:7b', 'mcq', 'B', 'B', 1),
    ('standard numeric passthrough',       'qwen2.5:7b', 'numeric_exact', '15', 'The answer is 15.\n#### 15', 1),
]

print(f"{'res':5} {'model':18} {'task':14} {'gold':5} | {'cleaned':40} {'guess':6} corr (exp)  -- label")
print('-'*135)
all_pass = True
for label, model, task, gold, raw, expected in cases:
    cleaned = clean_reasoning_response(raw) if is_reasoning_model(model) else raw
    correct, guess = score(task, gold, cleaned)
    ok = (correct == expected)
    all_pass &= ok
    print(f"{('OK' if ok else 'FAIL'):5} {model:18} {task:14} {gold:5} | {repr(cleaned[:38]):40} {guess:6} {correct}    ({expected})    -- {label}")
print()
print('ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED')

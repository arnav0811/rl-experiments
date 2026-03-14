from math_verify import parse, verify
def extract_answer(model_response):
    idx = model_response.rfind(r'\boxed{')
    if idx == -1:
        return None
    start = idx + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(model_response) and depth > 0:
        if model_response[i] == '{':
            depth += 1
        elif model_response[i] == '}':
            depth -= 1
        i += 1
    
    if depth != 0:
        return None

    model_answer = model_response[start:i - 1]
    return model_answer

def check_answer(model_answer, ground_truth):
    model_answer = extract_answer(model_answer)
    if model_answer is None:
        return False
    return verify(parse(model_answer), parse(ground_truth))



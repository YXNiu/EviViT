"""Versioned binary outcome contract; legacy experiment rewards are untouched."""
import json
from evivit_core.evivit_grpo import semantic_reward_judge_batch

def parse_answer(raw):
    value=str(raw).strip()
    if value.startswith('```'):
        lines=value.splitlines()
        if len(lines)>=3 and lines[-1].strip()=='```':
            value='\n'.join(lines[1:-1]).strip()
    if not value:return '', 'empty_answer'
    try: payload=json.loads(value)
    except json.JSONDecodeError:
        if value.startswith(('{','[')):return '', 'malformed_json'
        return value,None
    if isinstance(payload,dict):payload=payload.get('answer')
    if payload is None or isinstance(payload,(dict,list)):return '', 'invalid_answer'
    result=str(payload).strip()
    return (result,None) if result else ('','empty_answer')

def binary_reward(source):
    if source=='semantic_judge_error_fallback_wrong':
        raise RuntimeError('Judge failure is not a negative reward')
    return float(source in {'exact','relaxed','semantic_judge'})

def reliable_judge(tokenizer,model,**kwargs):
    """Retry malformed decisions singly with more output room; fail closed."""
    results=semantic_reward_judge_batch(tokenizer,model,**kwargs)
    for i,result in enumerate(results):
        if not result['judge_error']:continue
        retry=dict(kwargs,candidates=[result['candidate']],max_new_tokens=32)
        retried=semantic_reward_judge_batch(tokenizer,model,**retry)[0]
        retried['retried_after_raw']=result['raw_judge']
        if retried['judge_error']:raise RuntimeError('Unparseable reward judge after retry')
        results[i]=retried
    return results

def active_group(rewards):
    return bool(rewards) and max(rewards)!=min(rewards)

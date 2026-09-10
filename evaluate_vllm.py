import argparse
import ast
import concurrent.futures
import json
import operator
import re
import time
import urllib.request
from pathlib import Path

from prepare_data import SYSTEM, TOOLS

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow,ast.USub:operator.neg,ast.UAdd:operator.pos}
FINAL_RE=re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE=re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

def calc(expr):
    def ev(n):
        if isinstance(n,ast.Expression): return ev(n.body)
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
        if isinstance(n,ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.left),ev(n.right))
        if isinstance(n,ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsafe expression")
    value=ev(ast.parse(expr,mode="eval")); return str(int(value)) if isinstance(value,float) and value.is_integer() else str(value)

def post(url,payload):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read())

def one(args,row):
    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":row["question"]}]; calls=[]; errors=[]; final_text=""; start=time.time()
    for _ in range(args.max_turns):
        obj=post(args.url,{"model":args.model,"messages":messages,"tools":TOOLS,"tool_choice":"auto","temperature":args.temperature,"max_tokens":args.max_tokens,"chat_template_kwargs":{"enable_thinking":False}})
        m=obj["choices"][0]["message"]; final_text=m.get("content") or ""; tcs=m.get("tool_calls") or []
        messages.append({k:v for k,v in m.items() if k in ("role","content","tool_calls")})
        if not tcs: break
        for tc in tcs:
            try:
                raw=tc["function"]["arguments"]; a=json.loads(raw) if isinstance(raw,str) else raw; expr=a["expression"]; result=calc(expr); calls.append(expr)
                messages.append({"role":"tool","tool_call_id":tc.get("id"),"name":"calculator","content":result})
            except Exception as e:
                errors.append(str(e)); messages.append({"role":"tool","tool_call_id":tc.get("id"),"name":"calculator","content":"ERROR: invalid expression"})
    match=FINAL_RE.search(final_text); pred=match.group(1).replace(",","") if match else None; gold=str(row["answer"]).replace(",","")
    numbers=NUMBER_RE.findall(final_text); relaxed_pred=numbers[-1].replace(",","") if numbers else None
    try: correct=pred is not None and abs(float(pred)-float(gold))<1e-8
    except: correct=pred==gold
    try: relaxed_correct=relaxed_pred is not None and abs(float(relaxed_pred)-float(gold))<1e-8
    except: relaxed_correct=relaxed_pred==gold
    return {**row,"prediction":pred,"correct":correct,"relaxed_prediction":relaxed_pred,"relaxed_correct":relaxed_correct,"final_text":final_text,"tool_calls":calls,"tool_errors":errors,"repeat_calls":len(calls)-len(set(calls)),"latency":time.time()-start,"messages":messages}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--model",required=True); ap.add_argument("--url",default="http://127.0.0.1:8000/v1/chat/completions"); ap.add_argument("--concurrency",type=int,default=16); ap.add_argument("--temperature",type=float,default=0); ap.add_argument("--max-tokens",type=int,default=256); ap.add_argument("--max-turns",type=int,default=6); args=ap.parse_args()
    with args.data.open(encoding="utf-8") as handle:
        rows=[json.loads(x) for x in handle if x.strip()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool: results=list(pool.map(lambda r:one(args,r),rows))
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={"model":args.model,"n":len(results),"correct":sum(x["correct"] for x in results),"accuracy":sum(x["correct"] for x in results)/len(results),"relaxed_correct":sum(x["relaxed_correct"] for x in results),"relaxed_accuracy":sum(x["relaxed_correct"] for x in results)/len(results),"tool_use":sum(bool(x["tool_calls"]) for x in results),"tool_errors":sum(len(x["tool_errors"]) for x in results),"repeat_calls":sum(x["repeat_calls"] for x in results),"avg_latency":sum(x["latency"] for x in results)/len(results)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()

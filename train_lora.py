import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


class Decisions(Dataset):
    def __init__(self,path,tokenizer,max_length):
        self.items=[]
        with path.open(encoding="utf-8") as handle:
          for line in handle:
            if not line.strip(): continue
            row=json.loads(line); prompt=tokenizer.encode(row["prompt_text"],add_special_tokens=False); full=tokenizer.encode(row["full_text"],add_special_tokens=False)
            if len(full)<=max_length and full[:len(prompt)]==prompt and len(full)>len(prompt):
                self.items.append({"input_ids":torch.tensor(full),"labels":torch.tensor([-100]*len(prompt)+full[len(prompt):])})
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]


class Collator:
    def __init__(self,pad): self.pad=pad
    def __call__(self,items):
        ids=pad_sequence([x["input_ids"] for x in items],batch_first=True,padding_value=self.pad)
        labels=pad_sequence([x["labels"] for x in items],batch_first=True,padding_value=-100)
        return {"input_ids":ids,"labels":labels,"attention_mask":ids.ne(self.pad).long()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--epochs",type=float,default=1); ap.add_argument("--lr",type=float,default=2e-5); ap.add_argument("--rank",type=int,default=16); ap.add_argument("--max-length",type=int,default=1024); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args(); set_seed(args.seed)
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True); tok.pad_token=tok.eos_token
    ds=Decisions(args.data,tok,args.max_length)
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa")
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.rank*2,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    train_args=TrainingArguments(output_dir=str(args.output/"checkpoints"),per_device_train_batch_size=2,gradient_accumulation_steps=8,num_train_epochs=args.epochs,learning_rate=args.lr,lr_scheduler_type="cosine",warmup_ratio=.03,bf16=True,tf32=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},optim="adamw_torch_fused",logging_steps=5,save_strategy="no",report_to="none",remove_unused_columns=False,seed=args.seed)
    trainer=Trainer(model=model,args=train_args,train_dataset=ds,data_collator=Collator(tok.pad_token_id))
    start=time.time(); result=trainer.train(); wall=time.time()-start
    adapter=args.output/"adapter"; adapter.mkdir(parents=True,exist_ok=True); trainer.model.save_pretrained(adapter,safe_serialization=True); tok.save_pretrained(adapter)
    metrics={"samples":len(ds),"wall_seconds":wall,"train":result.metrics,"rank":args.rank,"lr":args.lr,"epochs":args.epochs}
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2))


if __name__=="__main__": main()

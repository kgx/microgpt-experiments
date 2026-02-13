# MicroGPT experiments

A minimal GPT in pure Python - no PyTorch, no dependencies. I built this to mess around and actually understand how the thing works. 

**Architecture:** [docs/architecture.md](docs/architecture.md)

**Credit:** Based on [Andrej Karpathy’s microgpt gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). What an amazing contribution.

---

```bash
 # train, saves model.json
python3 train.py             

# generate (no prompt)
python3 main.py               

# generate from prompt (use quotes if spaces)
python3 main.py -i "George"  
python3 main.py -i "Chanley"       
```
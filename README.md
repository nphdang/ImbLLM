# ImbLLM: Large Language Models for Tabular Imbalanced Classification
Paper: https://arxiv.org/abs/2510.09783
## Framework
![framework](https://github.com/nphdang/ImbLLM/blob/main/imbllm_method.jpg)
## Usage
Run ImbLLM method:
```bash
python -W ignore classify.py --dataset fuel --trainsize 1.0 --testsize 0.2 --g_encode original --c_encode ordinal --imbalance imb_llm --ratio 0.2 --runs 3
```
Run baselines:
```bash
run_classification.bat
```

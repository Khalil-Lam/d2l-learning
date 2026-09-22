# Tiny Tang Poetry LoRA

一个面向初学者的极小型微调实验：用 **Qwen3-0.6B** 做基础模型，通过 **LoRA** 学习“给定题目 -> 生成一首唐诗”。

目标不是做 SOTA，而是完整走通一次可审计的微调链：

`原始数据 -> train/dev/test -> baseline -> LoRA SFT -> held-out test -> 错误分析`

## 为什么选这个配置

- 基座：`Qwen/Qwen3-0.6B`，参数量小，适合第一次实验。
- 微调：LoRA，只训练少量适配参数，不改全部基础模型权重。
- 数据：公开 `chinese-poetry/chinese-poetry` 的《全唐诗》JSON；脚本只下载需要的几个分片，不把整库提交到本仓库。
- 任务：标题条件生成，而不是单纯续写。这样输入和输出边界清楚，适合理解 SFT。

## 目录

```text
tiny-tang-poetry-lora/
├─ prepare_data.py
├─ train_lora.py
├─ generate.py
├─ evaluate.py
├─ requirements.txt
└─ README.md
```

## 0. 建议环境

Windows + NVIDIA GPU 可以直接做。第一次先用 0.6B，不要急着上 3B/7B。

建议新建独立环境：

```powershell
conda create -n tang-poetry python=3.11 -y
conda activate tang-poetry
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

如果你的 CUDA/PyTorch 版本不同，以 PyTorch 官方安装命令为准。

## 1. 准备数据

在本目录运行：

```powershell
python prepare_data.py --files 3 --limit 1200
```

默认得到：

- train: 80%
- dev: 10%
- test: 10%

**test 不参与训练，也不用于选择 checkpoint。**

一条样本大致长这样：

```json
{
  "title": "秋夜",
  "prompt": "请写一首题为《秋夜》的唐诗：\n",
  "completion": "……"
}
```

## 2. 先跑 baseline

不要一上来训练。先看基础模型原本会写什么：

```powershell
python generate.py --title 秋夜
```

把输出保存下来。它是后面判断“微调到底改变了什么”的基线。

## 3. LoRA 微调

```powershell
python train_lora.py
```

默认配置：

- base model: Qwen3-0.6B
- LoRA rank: 8
- alpha: 16
- dropout: 0.05
- target: q/k/v/o projection
- batch size: 1
- gradient accumulation: 8
- lr: 2e-4
- epochs: 3
- max length: 256
- best checkpoint: 按 dev loss 选择

训练时，prompt token 的 label 被设为 `-100`，所以 loss **只计算唐诗答案部分**。

## 4. 微调后生成

```powershell
python generate.py --title 秋夜 --adapter outputs/tang-poetry-lora
```

重点不是“像不像李白”，而是比较：

1. 是否更像诗而不是普通说明文；
2. 行句是否更稳定；
3. 是否更容易围绕题目；
4. 是否出现机械重复；
5. 是否直接背训练集。

## 5. held-out test 比较

```powershell
python evaluate.py --adapter outputs/tang-poetry-lora --n 20
```

输出：

```text
runs/comparison.json
```

每个测试题同时保存：

- reference
- baseline
- LoRA
- 简单格式统计
- 是否与训练诗完全一致

注意：这些自动统计只是诊断工具，**不是“诗歌质量”的最终评分**。

## 第一次实验应该记录什么

至少保存：

```text
dataset version / source
train-dev-test counts
base model
seed
LoRA r / alpha / target modules
learning rate
epochs
best dev loss
20 个 test 的 baseline / LoRA 输出
失败案例
GPU 与运行时间
```

## 第一次实验的停止条件

如果下面四件事都成立，就算第一阶段完成：

- [ ] 训练能完整跑完
- [ ] train loss 正常下降
- [ ] dev loss 没有持续恶化
- [ ] LoRA 输出相对 baseline 有可观察变化

不要为了“分数更漂亮”反复看 test 后改参数。要调参，只看 train/dev。

## 第二阶段可以做什么

完成第一轮以后，再逐个做小实验：

1. 1200 首 -> 3000 首，观察收益；
2. r=8 vs r=16；
3. 只训 q/v vs q/k/v/o；
4. 标题条件 vs “主题+体裁”条件；
5. 加一个五言/七言格式控制字段；
6. 人工盲评 30 个输出。

一次只改一个因素，才能知道性能变化来自哪里。

## 数据与模型来源

本项目不重新分发《全唐诗》完整数据。运行 `prepare_data.py` 时从公开数据源读取需要的 JSON 分片。使用前请自行核对上游数据仓库的许可与数据说明。

基础模型请遵守其模型卡与许可证。

# WSL 专用 GPU 迁移包

仅面向 Windows 上的 WSL2 Linux x86_64 + NVIDIA GPU。请在 WSL 终端内执行，不支持 Windows 原生 Python、PowerShell 或 macOS。

要求：WSL2 内安装 Python 3.12；Windows 主机安装支持 WSL GPU 和 CUDA 12.8 的 NVIDIA 驱动；WSL 内 `nvidia-smi` 能识别显卡。安装依赖和首次下载模型需要联网。

解压后，在 WSL 终端进入 MedNLP_Research 目录：

```bash
bash setup-wsl.sh
.venv/bin/python e1_head_only.py
```

若 Python 3.12 路径不同，可执行 `PYTHON_BIN=/path/to/python3.12 bash setup-wsl.sh`。

安装器会创建本机 .venv，安装 PyTorch 2.9.0+cu128、Transformers 5.3.0 及锁定版本依赖，并验证 GPU 前向/反向计算。无需单独安装 CUDA Toolkit。首次安装需下载数 GB 依赖。

包内包含代码快照、原始/预处理/分词数据、依赖快照；不含旧电脑的虚拟环境、模型缓存或训练输出。不要直接复制旧电脑的 .venv 路径。

两台电脑同时直接运行会各自执行完整实验，不会自动分配任务或合并结果。

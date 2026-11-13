# 开始使用（简体中文，简明版）

Lobster 是装在你自己电脑上的生物信息分析工具。你用普通话说要做什么，它在本机运行，并写出一份 Jupyter 笔记。

English: [README.md](README.md)

## 安装

macOS 或 Linux：

```bash
curl -fsSL https://install.lobsterbio.com | bash
```

或者：

```bash
uv tool install 'lobster-ai[full]'
# 或 pip install 'lobster-ai[full]'
```

Windows（PowerShell）：

```powershell
irm https://install.lobsterbio.com/windows | iex
```

## 选择语言模型

```bash
lobster init
```

按提示选择模型。本机 Ollama 时，提示不会离开这台电脑。用云端模型时，提示会发给那个服务。

界面语言：

```bash
lobster --lang zh --help
```

也可以把 `LOBSTER_LANG=zh` 写进环境变量，或在 `.lobster_workspace/provider_config.json` 里加 `"lang": "zh"`。不认识的语言会回到英文。命令名和 `--json` 的字段名保持英文。

## 第一次分析

```bash
lobster chat
```

然后用中文描述你的数据和分析。例如：读取当前目录里的 h5ad，做质控、聚类和细胞类型注释，并在旁边写出 notebook。

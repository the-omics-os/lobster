# Lobster AI（简体中文，简明版）

Lobster AI 是开源的生物信息多智能体系统。你用平常的话描述要做的分析：各个组学领域的专家智能体会查找并读取数据，用 Scanpy、PyDESeq2 等科学软件库完成分析，并记录分析步骤；这些步骤可以导出为 Jupyter 笔记。它在你自己的电脑上运行，可以在它自己的终端对话里用，也可以通过 MCP 从你的编程助手里调用。

English: [README.md](README.md) · 完整文档（英文）：[docs.omics-os.com](https://docs.omics-os.com)

## 1. 安装

macOS 或 Linux：

```bash
curl -fsSL https://install.lobsterbio.com -o install.sh && bash install.sh
```

安装程序会问几个问题，所以要先下载再运行：`curl | bash` 这种管道方式读不到你的回答。

Windows：打开 PowerShell 或命令提示符，粘贴下面这一行并回车。完成后关闭窗口，再打开一个新窗口，输入 `lobster`。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://install.lobsterbio.com/windows | iex"
```

用 `uv` 或 `pip` 安装、之后添加分析包或升级，请看[安装指南](https://docs.omics-os.com/docs/getting-started/installation)。

## 2. 建一个项目

```bash
lobster init ~/my-study    # 家目录下的任意文件夹
cd ~/my-study
```

`lobster init` 会问你用哪个大语言模型服务，以及需要哪些分析包。项目把这些设置保存在数据和运行结果旁边，所以在这个文件夹里的对话、运行和 MCP 调用都用同一套设置。详见 [Lobster projects](https://docs.omics-os.com/docs/guides/lobster-projects)。

可选的服务：Omics-OS Cloud（默认）、Anthropic、AWS Bedrock、Google Gemini、OpenAI、OpenRouter、Nebius AI Studio、Azure AI、Ollama Cloud，或本机的 Ollama。用本机 Ollama 时，对话内容不会发给任何模型服务（在线数据库检索仍然需要联网）。用云端服务时，对话内容和工具结果会发给那个服务。

## 3. 用平常的话提问

```bash
lobster chat
```

然后用中文描述你的数据和分析。例如：读取当前目录里的 h5ad，做质控、聚类和细胞类型注释，最后导出记录下来的分析步骤。

## 界面语言

```bash
lobster --lang zh --help
```

也可以设置环境变量 `LOBSTER_LANG=zh`，或在 `.lobster_workspace/provider_config.json` 里加 `"lang": "zh"`。不认识的语言会回到英文。命令名和 `--json` 的字段名保持英文。

## 从编程助手里用 Lobster（MCP）

`lobster mcp --stdio` 把 Lobster 作为 MCP 服务器运行。Claude Code、Claude Desktop、Codex、Cursor 或 OpenCode 可以把一个实验问题交给它，拿回结果、笔记和文件。在 Claude Code 里，先进入你的项目文件夹：

```bash
claude mcp add --transport stdio lobster -- lobster mcp --stdio
```

其他助手的配置见 [MCP 指南](https://docs.omics-os.com/docs/guides/lobster-mcp)。

## 笔记、复跑和数据目录

- **导出：** 在 `lobster chat` 里，`/pipeline export` 把记录下来的分析步骤写成 Jupyter 笔记。
- **核对：** `lobster replay --verify analysis.ipynb --input my_data.h5ad` 重新运行笔记，并把它写出的文件和会话保存的文件作比较。它能发现它看得到的差异，但不能证明结果是怎样算出来的。详见 [replay and verify](https://docs.omics-os.com/docs/guides/replay-verify)。
- **数据目录（可选，默认关闭）：** 用 `lobster init ~/my-study --lamin` 时，Lobster 会把会话、你保存的数据集（含版本和内容哈希）以及导出的笔记记录在工作区里的本地 [LaminDB](https://lamin.ai) 目录中。

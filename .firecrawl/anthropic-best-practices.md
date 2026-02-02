[Skip to main content](https://code.claude.com/docs#content-area)

[Claude Code Docs home page![light logo](https://mintcdn.com/claude-code/o69F7a6qoW9vboof/logo/light.svg?fit=max&auto=format&n=o69F7a6qoW9vboof&q=85&s=536eade682636e84231afce2577f9509)![dark logo](https://mintcdn.com/claude-code/o69F7a6qoW9vboof/logo/dark.svg?fit=max&auto=format&n=o69F7a6qoW9vboof&q=85&s=0766b3221061e80143e9f300733e640b)](https://code.claude.com/docs)

![US](https://d3gk2c5xim1je2.cloudfront.net/flags/US.svg)

English

Search...

Ctrl KAsk AI

Search...

Navigation

Getting started

Claude Code overview

[Getting started](https://code.claude.com/docs/en/overview) [Build with Claude Code](https://code.claude.com/docs/en/sub-agents) [Deployment](https://code.claude.com/docs/en/third-party-integrations) [Administration](https://code.claude.com/docs/en/setup) [Configuration](https://code.claude.com/docs/en/settings) [Reference](https://code.claude.com/docs/en/cli-reference) [Resources](https://code.claude.com/docs/en/legal-and-compliance)

On this page

- [Get started in 30 seconds](https://code.claude.com/docs#get-started-in-30-seconds)
- [What Claude Code does for you](https://code.claude.com/docs#what-claude-code-does-for-you)
- [Why developers love Claude Code](https://code.claude.com/docs#why-developers-love-claude-code)
- [Use Claude Code everywhere](https://code.claude.com/docs#use-claude-code-everywhere)
- [Next steps](https://code.claude.com/docs#next-steps)
- [Additional resources](https://code.claude.com/docs#additional-resources)

## [​](https://code.claude.com/docs\#get-started-in-30-seconds)  Get started in 30 seconds

Prerequisites:

- A [Claude subscription](https://claude.com/pricing) (Pro, Max, Teams, or Enterprise) or [Claude Console](https://console.anthropic.com/) account

**Install Claude Code:**To install Claude Code, use one of the following methods:

- Native Install (Recommended)

- Homebrew

- WinGet


**macOS, Linux, WSL:**

Copy

Ask AI

```
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell:**

Copy

Ask AI

```
irm https://claude.ai/install.ps1 | iex
```

**Windows CMD:**

Copy

Ask AI

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

Native installations automatically update in the background to keep you on the latest version.

Copy

Ask AI

```
brew install --cask claude-code
```

Homebrew installations do not auto-update. Run `brew upgrade claude-code` periodically to get the latest features and security fixes.

Copy

Ask AI

```
winget install Anthropic.ClaudeCode
```

WinGet installations do not auto-update. Run `winget upgrade Anthropic.ClaudeCode` periodically to get the latest features and security fixes.

**Start using Claude Code:**

Copy

Ask AI

```
cd your-project
claude
```

You’ll be prompted to log in on first use. That’s it! [Continue with Quickstart (5 minutes) →](https://code.claude.com/docs/en/quickstart)

See [advanced setup](https://code.claude.com/docs/en/setup) for installation options, manual updates, or uninstallation instructions. Visit [troubleshooting](https://code.claude.com/docs/en/troubleshooting) if you hit issues.

## [​](https://code.claude.com/docs\#what-claude-code-does-for-you)  What Claude Code does for you

- **Build features from descriptions**: Tell Claude what you want to build in plain English. It will make a plan, write the code, and ensure it works.
- **Debug and fix issues**: Describe a bug or paste an error message. Claude Code will analyze your codebase, identify the problem, and implement a fix.
- **Navigate any codebase**: Ask anything about your team’s codebase, and get a thoughtful answer back. Claude Code maintains awareness of your entire project structure, can find up-to-date information from the web, and with [MCP](https://code.claude.com/docs/en/mcp) can pull from external data sources like Google Drive, Figma, and Slack.
- **Automate tedious tasks**: Fix fiddly lint issues, resolve merge conflicts, and write release notes. Do all this in a single command from your developer machines, or automatically in CI.

## [​](https://code.claude.com/docs\#why-developers-love-claude-code)  Why developers love Claude Code

- **Works in your terminal**: Not another chat window. Not another IDE. Claude Code meets you where you already work, with the tools you already love.
- **Takes action**: Claude Code can directly edit files, run commands, and create commits. Need more? [MCP](https://code.claude.com/docs/en/mcp) lets Claude read your design docs in Google Drive, update your tickets in Jira, or use _your_ custom developer tooling.
- **Unix philosophy**: Claude Code is composable and scriptable. `tail -f app.log | claude -p "Slack me if you see any anomalies appear in this log stream"` _works_. Your CI can run `claude -p "If there are new text strings, translate them into French and raise a PR for @lang-fr-team to review"`.
- **Enterprise-ready**: Use the Claude API, or host on AWS or GCP. Enterprise-grade [security](https://code.claude.com/docs/en/security), [privacy](https://code.claude.com/docs/en/data-usage), and [compliance](https://trust.anthropic.com/) is built-in.

## [​](https://code.claude.com/docs\#use-claude-code-everywhere)  Use Claude Code everywhere

Claude Code works across your development environment: in your terminal, in your IDE, in the cloud, and in Slack.

- **[Terminal (CLI)](https://code.claude.com/docs/en/quickstart)**: the core Claude Code experience. Run `claude` in any terminal to start coding.
- **[Claude Code on the web](https://code.claude.com/docs/en/claude-code-on-the-web)**: use Claude Code from your browser at [claude.ai/code](https://claude.ai/code) or the Claude iOS app, with no local setup required. Run tasks in parallel, work on repos you don’t have locally, and review changes in a built-in diff view.
- **[Desktop app](https://code.claude.com/docs/en/desktop)**: a standalone application with diff review, parallel sessions via git worktrees, and the ability to launch cloud sessions.
- **[VS Code](https://code.claude.com/docs/en/vs-code)**: a native extension with inline diffs, @-mentions, and plan review.
- **[JetBrains IDEs](https://code.claude.com/docs/en/jetbrains)**: a plugin for IntelliJ IDEA, PyCharm, WebStorm, and other JetBrains IDEs with IDE diff viewing and context sharing.
- **[GitHub Actions](https://code.claude.com/docs/en/github-actions)**: automate code review, issue triage, and other workflows in CI/CD with `@claude` mentions.
- **[GitLab CI/CD](https://code.claude.com/docs/en/gitlab-ci-cd)**: event-driven automation for GitLab merge requests and issues.
- **[Slack](https://code.claude.com/docs/en/slack)**: mention Claude in Slack to route coding tasks to Claude Code on the web and get PRs back.
- **[Chrome](https://code.claude.com/docs/en/chrome)**: connect Claude Code to your browser for live debugging, design verification, and web app testing.

## [​](https://code.claude.com/docs\#next-steps)  Next steps

[**Quickstart** \\
\\
See Claude Code in action with practical examples](https://code.claude.com/docs/en/quickstart) [**Common workflows** \\
\\
Step-by-step guides for common workflows](https://code.claude.com/docs/en/common-workflows) [**Troubleshooting** \\
\\
Solutions for common issues with Claude Code](https://code.claude.com/docs/en/troubleshooting) [**Desktop app** \\
\\
Run Claude Code as a standalone application](https://code.claude.com/docs/en/desktop)

## [​](https://code.claude.com/docs\#additional-resources)  Additional resources

[**About Claude Code** \\
\\
Learn more about Claude Code on claude.com](https://claude.com/product/claude-code) [**Build with the Agent SDK** \\
\\
Create custom AI agents with the Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) [**Host on AWS or GCP** \\
\\
Configure Claude Code with Amazon Bedrock or Google Vertex AI](https://code.claude.com/docs/en/third-party-integrations) [**Settings** \\
\\
Customize Claude Code for your workflow](https://code.claude.com/docs/en/settings) [**Commands** \\
\\
Learn about CLI commands and controls](https://code.claude.com/docs/en/cli-reference) [**Reference implementation** \\
\\
Clone our development container reference implementation](https://github.com/anthropics/claude-code/tree/main/.devcontainer) [**Security** \\
\\
Discover Claude Code’s safeguards and best practices for safe usage](https://code.claude.com/docs/en/security) [**Privacy and data usage** \\
\\
Understand how Claude Code handles your data](https://code.claude.com/docs/en/data-usage)

Was this page helpful?

YesNo

[Quickstart](https://code.claude.com/docs/en/quickstart)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.
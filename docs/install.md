# Install Instructions

You can either install through Pypi:

```bash
pip install compileiq
```

Or, build from the [repo](https://github.com/NVIDIA/CompileIQ) yourself:

```bash
pip install -e .
```

## Managing Searches with Claude Code

[Claude Code](https://claude.ai/code) can be used as an AI-assisted interface to manage CompileIQ searches interactively. This repository ships a set of slash commands under `.claude/commands/` that cover the full optimization workflow:

| Command | Description |
| --- | --- |
| `/compileiq-bootstrap` | Full stack setup — validates environment, installs dependencies, and configures frameworks |
| `/compileiq-code` | Generate optimization scripts with properly structured objective functions |
| `/compileiq-run` | Execute an optimization with GPU management, Ray setup, and progress monitoring |
| `/compileiq-optimize` | End-to-end pipeline orchestrator that chains all agents |
| `/compileiq-validate` | Benchmark top solutions against baselines with statistical rigor |
| `/compileiq-report` | Generate comprehensive reports with results and reproduction instructions |
| `/compileiq-configs` | Extract and manage optimized configurations from results |
| `/compileiq-integrate` | Find PTX-to-CUBIN compilation paths and generate integration code |
| `/compileiq-profile` | Deep kernel analysis using NSight Compute |
| `/compileiq-debug` | Diagnose convergence issues, invalid configs, and ptxas errors |

Once Claude Code is installed, clone the repository and these commands will be available automatically in any Claude Code session opened from the project root.

## Environment Configuration Options

| Environment Variable | Default Value | Type | Description
| ------ | ------ | ------ | ------ |
| CIQ_SOCKET_TIMEOUT | 20 | int | Controls how long Solar waits for a core response. If you experience timeouts because your search space is too big, consider increasing this value.
| CIQ_KEEP_CACHE | False | bool | If set to True, `.cache` files will not be deleted.
| CIQ_PROCESS_MODE | "forkserver" | str | User can set this to "fork" to better separate processes and deal with threads. `IsoMultiProcessWorker` uses `spawn` as the default.

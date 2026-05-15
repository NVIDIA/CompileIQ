# CompileIQ - NVIDIA's Official Compiler HPO

CompileIQ is an evolutionary-based hyper-parameter optimizer tailored to tune our internal compiler controls.

## Quick install

You can either install through Pypi:

```bash
pip install compileiq
```

Or, build from the [repo](https://github.com/NVIDIA/CompileIQ) yourself:

```bash
pip install -e .
```

## Search Spaces

CompileIQ can retrieve curated compiler search spaces from GitHub release assets and cache them locally. Use `PtxasSearchSpace` or `NvccSearchSpace` to select a compiler, compiler version, and optional variant:

```python
from compileiq.search_spaces.compilers import PtxasSearchSpace

search_space = PtxasSearchSpace(version="13.3", variant="att")
```

For reproducible runs, pin a search-space release tag:

```python
search_space = PtxasSearchSpace(version="13.3", tag="search-spaces-2026.05.05")
```

Set `CIQ_SEARCH_SPACES_DIR` to use a local mirror containing `manifest.json` plus the referenced `.bin` files. Set `CIQ_SEARCH_SPACES_REPO` to test or use a different release repository.

## Environment Configuration Options

| Environment Variable | Default Value | Type | Description
| ------ | ------ | ------ | ------ |
| CIQ_SOCKET_TIMEOUT | 20 | int | Controls how long CompileIQ waits for a core response. If you experience timeouts because your search space is too big, consider increasing this value.
| CIQ_KEEP_CACHE | False | bool | If set to True, `.cache` files will not be deleted.
| CIQ_PROCESS_MODE | "forkserver" | str | Start method for process-based workers. Set to "fork" for tighter process separation when threads are involved. `IsoMultiProcessWorker` defaults to "fork" independently.
| CIQ_SEARCH_SPACES_DIR | unset | path | Reads compiler search-space `manifest.json` and `.bin` files from a local mirror instead of GitHub.
| CIQ_SEARCH_SPACES_REPO | NVIDIA/CompileIQ | str | GitHub repository used for search-space release lookups, useful for staging or a future dedicated asset repo.

## Examples

The `examples/` folder has simple examples for you to get started on using CompileIQ.

If you are planning on running examples, you may need additional dependencies:

```bash
python -m poetry install --with examples
```

## Documentation development

Install the docs dependencies once:

```bash
make install-docs
```

To preview uncommitted documentation edits from your live worktree:

```bash
make docs-preview
```

Then open <http://localhost:8000/main/>.

If port 8000 is already in use, stop the existing local docs server first.

Use `make docs` or `make docs-serve` when you need to test the multiversion
documentation shape used by GitHub Pages. Those commands build from Git refs, so
they are not the right choice for checking dirty worktree edits before commit.

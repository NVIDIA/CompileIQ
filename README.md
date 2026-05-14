# CompileIQ - Nvidia's Official Compiler HPO

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

We provide curated search spaces under the `assets/` folder. These can be fed directly into CompileIQ or mixed with user-space for a co-tuning scenario.

## Environment Configuration Options

| Environment Variable | Default Value | Type | Description
| ------ | ------ | ------ | ------ |
| CIQ_SOCKET_TIMEOUT | 20 | int | Controls how long CompileIQ waits for a core response. If you experience timeouts because your search space is too big, consider increasing this value.
| CIQ_KEEP_CACHE | False | bool | If set to True, `.cache` files will not be deleted.
| CIQ_PROCESS_MODE | "forkserver" | str | Start method for process-based workers. Set to "fork" for tighter process separation when threads are involved. `IsoMultiProcessWorker` defaults to "fork" independently.

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

# Draft issue for 54yyyu/kaggle-mcp

**Note:** Claude's `gh` CLI auth doesn't have `public_repo` scope, so the issue
couldn't be auto-filed. Open https://github.com/54yyyu/kaggle-mcp/issues/new
and paste the content below.

---

**Title:** Two install-blocking bugs: empty wheel (missing packages.find) + obsolete FastMCP kwarg

**Body:**

Hi — just installed `kaggle-mcp` and hit two separate bugs that both prevent the server from starting. Filing together since they're simple one-line fixes.

## Bug 1 — Empty wheel (missing `[tool.setuptools.packages.find]`)

`pyproject.toml` declares `package-dir = {"" = "src"}` but is missing the corresponding `[tool.setuptools.packages.find]` section. Without it, setuptools doesn't know where to look for packages, and the resulting wheel ships only the `.dist-info` directory with no `kaggle_mcp/` Python source inside.

**Repro (Python 3.12, pip 24.x, setuptools 68+):**

```bash
pip install git+https://github.com/54yyyu/kaggle-mcp.git
python -c "import kaggle_mcp"
# ModuleNotFoundError: No module named 'kaggle_mcp'

kaggle-mcp
# Traceback (most recent call last):
#   File "...\Scripts\kaggle-mcp.exe\__main__.py", line 4, in <module>
#     from kaggle_mcp.server import main
# ModuleNotFoundError: No module named 'kaggle_mcp'
```

The built wheel is only ~4KB — you can confirm the source is missing by unzipping it.

**Fix:** add three lines to `pyproject.toml`:

```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]      # <-- NEW
where = ["src"]                      # <-- NEW
```

After this patch the wheel grows to ~22KB and `kaggle_mcp` imports cleanly.

## Bug 2 — `FastMCP.__init__()` got an unexpected keyword argument `description`

After fixing Bug 1 and trying to run `kaggle-mcp`:

```
Traceback (most recent call last):
  File "...\site-packages\kaggle_mcp\server.py", line 22, in <module>
    mcp = FastMCP("Kaggle", description="Kaggle API integration through the Model Context Protocol")
TypeError: FastMCP.__init__() got an unexpected keyword argument 'description'
```

Current `mcp>=1.3.0` (and certainly `mcp[cli]>=1.3.0` as pinned in `pyproject.toml`'s dependencies) does not accept a `description` kwarg on `FastMCP`. The signature only takes a single positional name argument.

**Fix** — `src/kaggle_mcp/server.py` line 22:

```python
# before
mcp = FastMCP("Kaggle", description="Kaggle API integration through the Model Context Protocol")

# after
mcp = FastMCP("Kaggle")
```

## Environment

- Windows 11, Python 3.12.10, pip 24.x
- Installed via `pip install git+https://...` and also via `uvx --from git+... kaggle-mcp` — both hit Bug 1
- Installed into a local venv; verified both bugs and both fixes applied locally
- After both patches: `from kaggle_mcp.server import mcp` imports cleanly, the server starts and waits for stdio input as expected

Happy to send a PR if you'd like.

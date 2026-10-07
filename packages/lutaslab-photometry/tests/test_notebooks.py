import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_all_notebook_code_cells_compile(self):
        folder = Path("packages/lutaslab-photometry/examples/notebooks")
        for path in folder.glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")

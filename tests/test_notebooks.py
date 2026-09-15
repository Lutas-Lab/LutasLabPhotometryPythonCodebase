import json
from pathlib import Path
import unittest


class NotebookTests(unittest.TestCase):
    def test_all_notebook_code_cells_compile(self):
        for path in Path("notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")

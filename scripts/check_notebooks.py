"""Check all release notebooks without starting kernels or downloading data."""
import ast
import hashlib
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

root = Path(__file__).resolve().parents[1]
transformer = TransformerManager()
notebooks = sorted(root.glob('Tutorial_*/*.ipynb'))
assert len(notebooks) == 15, 'Expected fifteen released notebooks'
for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert cells and all(not cell.outputs and cell.execution_count is None for cell in cells)
    for index, cell in enumerate(cells, 1):
        ast.parse(transformer.transform_cell(cell.source), filename=f'{path.name}:code-cell-{index}')
    relative = path.relative_to(root).as_posix()
    url = 'https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/' + relative
    assert url in (root / 'README.md').read_text(encoding='utf-8'), f'Missing index badge: {relative}'
    assert any(url in cell.source for cell in notebook.cells if cell.cell_type == 'markdown')
    digest = hashlib.sha256('\n\n'.join(cell.source for cell in cells).encode()).hexdigest()
    print(f'{relative}: {len(cells)} code cells; SHA256 {digest}')
print('PASS: format, unique IDs, cleared outputs, Python/IPython syntax, and Colab targets.')

import py_compile
import tempfile
import os
from typing import Dict

def syntax_check(code: str) -> Dict:
    """
    Write `code` to a temp file and run py_compile to check syntax only.
    Returns dict with success bool and message.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            py_compile.compile(path, doraise=True)
            return {"ok": True, "message": "Syntax OK"}
        except py_compile.PyCompileError as e:
            return {"ok": False, "message": str(e)}

import ast
from pathlib import Path
from typing import Dict, List
import unittest

ROOT = Path(__file__).resolve().parents[1]
def load_functions(file, names, scope):
    tree = ast.parse((ROOT/file).read_text())
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),scope)
    return scope

class LocalLogicTest(unittest.TestCase):
    def test_report_excludes_outside_dates(self):
        data=[{'date':'2026-01-01','subject':'физика','url':'https://example.com/a'}, {'date':'2026-02-01','subject':'физика','url':'https://example.com/b'}]
        scope={'List':List,'Dict':Dict,'read_all':lambda:data,'SUBJECTS':['физика']}
        load_functions('collect_cli.py',{'normalize_subject','build_report'},scope)
        self.assertEqual(scope['build_report']('все','2026-01-01','2026-01-31'),{'физика':[data[0]]})

    def test_chunks_preserve_overlap(self):
        scope=load_functions('search.py',{'chunks'}, {})
        text=''.join(chr(1000+i) for i in range(30))
        parts=list(scope['chunks'](text,n=10,ov=2))
        self.assertEqual(parts[0][-2:],parts[1][:2])
        self.assertEqual(parts[-1],text[24:])

if __name__ == '__main__':
    unittest.main()

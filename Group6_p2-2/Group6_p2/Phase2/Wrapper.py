import os, sys, runpy

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

runpy.run_path(os.path.join(_dir, "nerf.py"), run_name="__main__")

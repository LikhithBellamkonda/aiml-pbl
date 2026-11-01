import sys
import pathlib
# Add the project’s src and dashboard packages to PYTHONPATH for test imports
# This file lives in <workspace>/aiml-pbl/tests, so two parents up gives the workspace root
repo_root = pathlib.Path(__file__).resolve().parents[2]
src_path = repo_root / "aiml-pbl" / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

dash_path = repo_root / "aiml-pbl" / "dashboard"
if str(dash_path) not in sys.path:
    sys.path.append(str(dash_path))

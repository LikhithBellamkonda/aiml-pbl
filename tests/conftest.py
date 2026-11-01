import sys
import pathlib
# Add the project’s src and dashboard packages to PYTHONPATH for test imports
# This file lives in <workspace>/aiml-pbl/tests, so two parents up gives the workspace root
repo_root = pathlib.Path(__file__).resolve().parents[2]
# Add the directory that contains the 'src' package (aiml-pbl) to sys.path
project_root = repo_root / "aiml-pbl"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# Dashboard lives alongside src under the same project root
dash_path = project_root / "dashboard"
if str(dash_path) not in sys.path:
    sys.path.append(str(dash_path))

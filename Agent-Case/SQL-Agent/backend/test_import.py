import sys, traceback, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    from app.orchestrator.orch_exec import Orchestrator
    print('IMPORT OK')
except Exception:
    traceback.print_exc()

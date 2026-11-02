"""Resource-aware camera evaluation controller."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from publish import publish
from refcloth_eval_scheduler import entry
if __name__ == '__main__':
    entry('coupling',publish)

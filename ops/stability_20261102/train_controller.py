"""Resource-aware detached source controller."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refcloth_source_scheduler import entry
if __name__ == '__main__':
    entry('stability')

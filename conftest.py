"""Root conftest.
 
Belt-and-braces alongside pytest.ini's `pythonpath` setting: older pytest
versions (<7.0) ignore that option, and some IDE runners bypass pytest.ini
entirely. This guarantees `from src...` imports resolve either way.
"""
 
import sys
from pathlib import Path
 
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
 

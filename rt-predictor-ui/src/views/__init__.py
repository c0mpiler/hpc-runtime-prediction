# Pages module initialization
import sys
from pathlib import Path

# Add parent directory to path for utils imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

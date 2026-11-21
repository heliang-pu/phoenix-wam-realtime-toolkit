"""Upstream scripts/train.py (or tools/text_cache.py) with fmc3_phoenix.install() applied first.

python train_fmc3.py <phoenix checkout>/scripts/train.py task=fmc3_posttrain ...   # run from the checkout
"""

from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmc3_phoenix

fmc3_phoenix.install()
script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")

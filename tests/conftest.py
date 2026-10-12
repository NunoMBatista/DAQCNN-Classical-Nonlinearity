import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))


def pytest_configure(config):
    # list-valued categorical choices (topology lists) are intended
    config.addinivalue_line("filterwarnings", "ignore:Choices for a categorical distribution:UserWarning")

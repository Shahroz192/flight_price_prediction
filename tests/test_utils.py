import sys
from src.utils import add_project_root_to_path


def test_add_project_root_to_path():
    initial_path_len = len(sys.path)
    add_project_root_to_path()
    assert len(sys.path) >= initial_path_len

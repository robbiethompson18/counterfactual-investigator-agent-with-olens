from modname import module_name


def test_module_name():
    assert module_name("src/happy.py") == "happy"

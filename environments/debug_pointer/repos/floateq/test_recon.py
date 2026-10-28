from recon import reconciled


def test_reconciled():
    assert reconciled([0.1, 0.2], 0.3) is True

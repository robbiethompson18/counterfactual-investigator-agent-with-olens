from label import label_for


def test_label():
    assert label_for("7", [(7, "seven"), (8, "eight")]) == "seven"

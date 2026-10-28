from signup import register


def test_register():
    assert register("alice_wonder", set()) == "ok"

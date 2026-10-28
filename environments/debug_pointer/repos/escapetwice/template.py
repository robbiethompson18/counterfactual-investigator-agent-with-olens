from safe import escape


def h1(title):
    return f"<h1>{escape(title)}</h1>"

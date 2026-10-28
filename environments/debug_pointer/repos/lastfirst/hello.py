from greet import first_name
from people import full_name
from punct import exclaim


def hello(uid):
    return exclaim("Hi " + first_name(full_name(uid)))

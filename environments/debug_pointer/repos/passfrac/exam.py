from marks import score_fraction
from rule import passed
from words import verdict


def outcome(correct, total):
    return verdict(passed(score_fraction(correct, total)))

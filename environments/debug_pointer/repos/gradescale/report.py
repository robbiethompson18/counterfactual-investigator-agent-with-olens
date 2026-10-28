from curve import curve
from letter import letter
from score import score


def grade(correct, total):
    return letter(curve(score(correct, total)))

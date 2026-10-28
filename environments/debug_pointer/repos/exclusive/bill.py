from rate import charge
from stay import nights


def cost(checkin, checkout, nightly):
    return charge(nights(checkin, checkout), nightly)

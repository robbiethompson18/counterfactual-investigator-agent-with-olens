from dash import dash
from fmt import hh
from span import end_hour


def slot(start, hours):
    return dash(hh(start), hh(end_hour(start, hours)))

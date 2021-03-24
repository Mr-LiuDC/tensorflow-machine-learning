from flask import Blueprint

from . import views

mnist = Blueprint('mnist', __name__)

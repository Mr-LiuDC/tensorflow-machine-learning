from flask import Blueprint

mnist = Blueprint('mnist', __name__)

from . import views

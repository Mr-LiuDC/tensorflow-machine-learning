import os

from flask import Flask
from flask_bootstrap import Bootstrap

from app.config import app_config


def create_app(config_name='development'):
    app = Flask(__name__)
    if os.getenv('FLASK_CONFIG') == "production":
        app.config.update(
            SECRET_KEY=os.getenv('SECRET_KEY'),
            SQLALCHEMY_DATABASE_URI=os.getenv('SQLALCHEMY_DATABASE_URI')
        )
    else:
        app.config.from_object(app_config[config_name])
        app.config.from_pyfile('config.py')

    Bootstrap(app)

    from .mnist import mnist as mnist_blueprint
    app.register_blueprint(mnist_blueprint)

    return app
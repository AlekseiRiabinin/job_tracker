from flask import Flask
from . import routes


def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(routes.bp)
    
    return app

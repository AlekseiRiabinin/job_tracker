from flask import Flask
from flasgger import Swagger
from . import routes



def create_app():
    app = Flask(__name__)
    Swagger(app)
    
    app.register_blueprint(routes.bp)
    return app

from flask import Flask
from flasgger import Swagger

def create_app():
    app = Flask(__name__)
    Swagger(app)
    
    from . import routes
    app.register_blueprint(routes.bp)
    
    return app

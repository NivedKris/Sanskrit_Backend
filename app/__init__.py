from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

def create_app(test_config=None):
    # Create Flask app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    
    # Load environment variables
    load_dotenv()
    
    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
        
    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev'),
        DATABASE=os.path.join(app.instance_path, 'chroma.db'),
        UPLOAD_FOLDER=os.path.join(app.instance_path, 'uploads'),
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.update(test_config)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register blueprints
    from .routes import bp
    app.register_blueprint(bp)

    return app 
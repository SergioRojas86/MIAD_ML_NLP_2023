from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from flask_cors import CORS
from model_logistic_regression import prediccion_genero

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Predicción genero peliculas API',
    description='Predicción genero peliculas API')

ns = api.namespace('prediccion', 
     description='Predicción genero peliculas modelo')
   
parser = api.parser()

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Plot de la pelicula', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class MovieApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        return {
         "result": prediccion_genero([args['Plot']])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
#from Proyecto_1.model_XGB import predict_price 
from flask_cors import CORS
from model_LR import predict_genre

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Movie genre Prediction API',
    description='Movie genre Prediction API')

ns = api.namespace('predict', 
     description='Movie genre regressor')
   
parser = api.parser()

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Sinopsis de la pelicula', 
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
         "The movie genre is": predict_genre([args['Plot']])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
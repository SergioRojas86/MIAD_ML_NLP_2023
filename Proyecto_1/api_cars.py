from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
#from Proyecto_1.model_XGB import predict_price 
from flask_cors import CORS
from model_XGB import predict_price

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Car price Prediction API',
    description='Car price Prediction API')

ns = api.namespace('predict', 
     description='Car price regressor')
   
parser = api.parser()

parser.add_argument(
    'Año', 
    type=int, 
    required=True, 
    help='Año del modelo del carro', 
    location='args')

parser.add_argument(
    'Millaje', 
    type=int, 
    required=True, 
    help='Millas recorridas', 
    location='args')

parser.add_argument(
    'Estado', 
    type=str, 
    required=True, 
    help='Ubicación geográfica del carro', 
    location='args')

parser.add_argument(
    'Marca', 
    type=str, 
    required=True, 
    help='Marca del carro', 
    location='args')

parser.add_argument(
    'Modelo', 
    type=str, 
    required=True, 
    help='Modelo de la marca seleccionada', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class CarApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        return {
         "result": predict_price([args['Año'],args['Millaje'],args['Estado'],args['Marca'],args['Modelo']])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
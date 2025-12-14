import json
import base64
import boto3

# Using low-level client representing Amazon SageMaker Runtime ( To invoke endpoint)
runtime_client = boto3.client('sagemaker-runtime')

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2025-12-14-09-55-40-820"## TODO: fill in

#Not using the available template because sagemaker import is not available and not enough space in jupyter labs to create layer zip import
def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["body"]['image_data'])     ## TODO: fill in (Decoding the encoded 'Base64' image-data and class remains'bytes')

    # Instantiate a Predictor (Here we have renamed 'Predictor' to 'response')
    # Response after invoking a deployed endpoint via SageMaker Runtime 
    response = runtime_client.invoke_endpoint(
                                        EndpointName=ENDPOINT,    # Endpoint Name
                                        Body=image,               # Decoded Image Data as Input (class:'Bytes') Image Data
                                        ContentType='image/png'   # Type of inference input data - Content type  (Eliminatesthe need of serializer)
                                    )
                                    
    
    # Make a prediction: Unpack reponse
    # (NOTE: 'response' returns a dictionary with inferences in the "Body" : (StreamingBody needs to be read) having Content_type='string')
    
    ## TODO: fill in (Read and decode predictions/inferences to 'utf-8' & convert JSON string obj -> Python object)
    inferences = json.loads(response['Body'].read().decode('utf-8'))     # list
  
    
    # We return the data back to the Step Function    
    event["body"]['inferences'] = inferences            ## List of predictions               
    return {
        'statusCode': 200,
        'body': event["body"]                          ## Passing the event python dictionary in the body
    }
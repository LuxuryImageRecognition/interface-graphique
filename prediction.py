from google.cloud import automl

PROJECT_ID = "image-storage-350313"
MODEL_ID = "ICN8608983630951219200"


# 'content' is base-64-encoded image data.
def get_prediction(content):
    prediction_client = automl.PredictionServiceClient()

    # Get the full path of the model.
    model_full_id = automl.AutoMlClient.model_path(PROJECT_ID, "us-central1", MODEL_ID)

    image = automl.Image(image_bytes=content)
    payload = automl.ExamplePayload(image=image)

    # params is additional domain-specific parameters.
    # score_threshold is used to filter the result
    # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#predictrequest
    params = {"score_threshold": "0.6"}

    request = automl.PredictRequest(name=model_full_id, payload=payload, params=params)
    response = prediction_client.predict(request=request)
    return response

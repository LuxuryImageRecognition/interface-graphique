from google.cloud import automl_v1beta1


# 'content' is base-64-encoded image data.
def get_prediction(content, project_id, model_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()

    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content}}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned


with open("boxed_images.jpeg", 'rb') as ff:
    image = ff.read()

get_prediction(image, "image-storage-350313", "ICN8608983630951219200")

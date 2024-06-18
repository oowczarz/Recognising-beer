from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from inference_sdk import InferenceHTTPClient
import base64
from io import BytesIO


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=" "
)
ROBOFLOW_MODEL = 'beers-piex7/1'


@csrf_exempt
def main_view(request):
    if request.method == 'GET':
        return render(request, 'index.html')
    elif request.method == 'POST':
        beer_image = request.FILES['beer_image']
        image = Image.open(beer_image)
        result = CLIENT.infer(image, model_id=ROBOFLOW_MODEL)
        name = result['predictions'][0]['class']

        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode()

        context = {
            'name': name,
            'image_data': image_data,
        }
        return render(request, 'answer.html', context)

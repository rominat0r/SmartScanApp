from wsgiref.util import FileWrapper
from django.http import HttpResponse, JsonResponse
import requests
from .serializers import ProductSerializer
from  .models import ProductModel
from  rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
from rest_framework.views import APIView
import main
# Create your views here.

class ProductList(APIView): 
    #permission_classes = (IsAuthenticated, )
    def get(self, request):
        product = ProductModel.objects.all()
        serializer = ProductSerializer(product, many = True)
        return JsonResponse({'products':serializer.data})
    def post(self, request): 
        serializer = ProductSerializer(data = request.data)
        if serializer.is_valid():
            serializer.save()
            imgname = serializer.data['image'].rsplit('/', 1)[-1]
            result = main.scan(imgname)
            #result = requests.post('http://127.0.0.1:8000', json= {'image':str(imgname)})
            return JsonResponse({'products':result})
        else:
            return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
class SSLVerify(APIView):
    def get(self,request): 
        zip_file = open('/var/www/html/BackendUser/mediafiles/posts/392497235DD94748B6173D275B22FEA1.txt', 'rb')
        response = HttpResponse(FileWrapper(zip_file), content_type='application/txt')
        response['Content-Disposition'] = 'attachment; filename="%s"' % '392497235DD94748B6173D275B22FEA1.txt'
        return response

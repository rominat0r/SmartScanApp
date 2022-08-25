from rest_framework.permissions import AllowAny
from .serializers import RegisterSerializer
from rest_framework import generics

# Create your views here.
class RegisterUserAPIView(generics.CreateAPIView):
  permission_classes = (AllowAny,)
  serializer_class = RegisterSerializer

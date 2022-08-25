from pyexpat import model
from  product_scan.models  import ProductModel
from rest_framework import serializers

class ProductSerializer(serializers.ModelSerializer):
    class Meta: 
        model = ProductModel
        fields = ['id', 'image']

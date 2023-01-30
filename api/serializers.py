from dataclasses import fields
from pyexpat import model
from rest_framework import serializers
# from rest_framework.validators import UniqueTogetherXValidator

from .models import (
    customer_chat,
    SpreeQuestions,
    SpreeAnswerOptions,
    SpreeProducts,
    ApiProducts,
    customerquizresponse,
    ApiQuizquestionsNew,
    ApiIngredients,
    foundation_image,
    tryons_image
)


class recievemessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = customer_chat
        fields = ('question', 'sessid')


class recievereplySerializer(serializers.ModelSerializer):
    class Meta:
        model = customer_chat
        fields = ('question', 'sessid')


class QuizQuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpreeQuestions
        fields = '__all__'


class AnswerOptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpreeAnswerOptions
        fields = '__all__'


class chatRecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApiProducts
        fields = ('ingredient', 'function_ingredients', 'subtitle')


class customerResponseQuizSerializers(serializers.ModelSerializer):
    class Meta:
        model = customerquizresponse
        fields = ('spree_option_id', 'spree_question_id', 'question_text',
                  'option_text', 'important_ingredients', 'sess_id', "slug")


class ingredientSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApiQuizquestionsNew
        fields = '__all__'


class IngredientRecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = customerquizresponse
        fields = ('ingredient', 'function_ingredients', 'subtitle')


class allProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpreeProducts
        fields = ('name', )


class FoundationImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = foundation_image
        fields = "__all__"


class ExtractSkinSerializer(serializers.ModelSerializer):
    class Meta:
        model = foundation_image
        fields = "__all__"


#  Add Serializers for try ons

class TryOnsSerializers(serializers.ModelSerializer):
    class Meta:
        model = tryons_image
        fields = "__all__"


class TryonsImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = tryons_image
        fields = "__all__"

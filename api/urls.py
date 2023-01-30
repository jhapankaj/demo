from django.urls import path
# from django.conf.urls import url
from api import views

from rest_framework.urlpatterns import format_suffix_patterns

# from demo.api.views import ImageAPIViews
# from  import views

app_name = "api"

urlpatterns = [
    path('recievemessage',  views.recievemessageView.as_view()),
    # path('allproducts',  views.allProductViews.as_view()),
    # path('recievereply/<str:pk>/', views.recievereplyView.as_view()),
    path('quizquestion/<str:pk>/', views.QuizQuestionView.as_view()),
    path('answeroption/<int:pk>/', views.AnswerOptionView.as_view()),
    # path('chatrecommendation/<str:pk>/', views.chatRecommendationView.as_view()),
    path('customerresponsequiz', views.customerResponseQuizView.as_view()),
    path('ingredient/<int:pk>/', views.ingredientView.as_view()),
    path('recommendationskincare/<str:pk>/',
         views.IngredientRecommendationViewSkinCare.as_view()),
    path('product_search/<str:pk>/',
         views.product_es_searchView.as_view()),
    path('foundation_image_upload', views.FoundationImageUploadView.as_view()),
    path('foundation_matching_advisor/<int:pk>/',
         views.ExtractSkinViewFoundation.as_view()),

    path('id/<str:imageid>', views.ImageAPIViews.as_view()),
    path('tryons_image_lipstick', views.TryOnsImageLipstickView.as_view()),
    path('tryonsid/<str:sess_id>', views.TryonsImageAPIViews.as_view()),
    path('tryons_image_upload', views.TryonImageUploadView.as_view()),
]

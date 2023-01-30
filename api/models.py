from platform import mac_ver
from pyexpat import model
from django.db import models

# Create your models here.


class customer_chat(models.Model):
    sessid = models.TextField(null=True)
    question = models.TextField(null=True)
    response = models.TextField(null=True)
    pred_tag = models.TextField(null=True, blank=True)
    qid = models.IntegerField(null=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)


class SpreeQuestions(models.Model):
    id = models.BigIntegerField(primary_key=True)
    text = models.TextField(blank=True, null=True)
    quiz_type = models.IntegerField(blank=True, null=True)
    position = models.IntegerField(blank=True, null=True)
    multiple_answer = models.IntegerField(blank=True, null=True)
    created_at = models.TextField(blank=True, null=True)
    updated_at = models.TextField(blank=True, null=True)
    freetext = models.IntegerField(blank=True, null=True)
    answers_count = models.IntegerField(blank=True, null=True)
    unlimited = models.IntegerField(blank=True, null=True)
    slug = models.CharField(max_length=4, blank=True, null=True)
    extra_info = models.TextField(blank=True, null=True)
    is_alergen = models.IntegerField(blank=True, null=True)
    is_city = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'spree_questions'


class SpreeAnswerOptions(models.Model):
    id = models.BigIntegerField(primary_key=True)
    text = models.TextField(blank=True, null=True)
    position = models.IntegerField(blank=True, null=True)
    spree_question_id = models.BigIntegerField(blank=True, null=True)
    created_at = models.TextField(blank=True, null=True)
    updated_at = models.TextField(blank=True, null=True)
    description = models.CharField(max_length=487, blank=True, null=True)
    important_ingreds = models.TextField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'spree_answer_options'


class SpreeProducts(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=94, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    available_on = models.TextField(blank=True, null=True)
    deleted_at = models.TextField(blank=True, null=True)
    slug = models.CharField(max_length=107, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)
    meta_keywords = models.TextField(blank=True, null=True)
    tax_category_id = models.IntegerField(blank=True, null=True)
    shipping_category_id = models.IntegerField(blank=True, null=True)
    created_at = models.TextField(blank=True, null=True)
    updated_at = models.TextField(blank=True, null=True)
    promotionable = models.IntegerField(blank=True, null=True)
    meta_title = models.CharField(max_length=157, blank=True, null=True)
    discontinue_on = models.TextField(blank=True, null=True)
    vendor_id = models.IntegerField(blank=True, null=True)
    short_description = models.TextField(blank=True, null=True)
    how_to_use = models.TextField(blank=True, null=True)
    ingredients = models.TextField(blank=True, null=True)
    spree_brand_id = models.BigIntegerField(blank=True, null=True)
    spree_country_id = models.BigIntegerField(blank=True, null=True)
    why_we_love_it = models.TextField(blank=True, null=True)
    shipping_info = models.TextField(blank=True, null=True)
    good_to_know = models.TextField(blank=True, null=True)
    how_to_store = models.TextField(blank=True, null=True)
    avg_rating = models.DecimalField(
        max_digits=10, decimal_places=0, blank=True, null=True)
    reviews_count = models.IntegerField(blank=True, null=True)
    is_gift_card = models.IntegerField(blank=True, null=True)
    is_e_gift_card = models.IntegerField(blank=True, null=True)
    unique_identifier = models.TextField(blank=True, null=True)
    unique_identifier_type = models.CharField(
        max_length=4, blank=True, null=True)
    feed_active = models.IntegerField(blank=True, null=True)
    jetti_id = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'spree_products'


class ApiProducts(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    available_on = models.TextField(blank=True, null=True)
    deleted_at = models.TextField(blank=True, null=True)
    slug = models.TextField(blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)
    meta_keywords = models.TextField(blank=True, null=True)
    tax_category_id = models.IntegerField(blank=True, null=True)
    shipping_category_id = models.IntegerField(blank=True, null=True)
    created_at = models.TextField(blank=True, null=True)
    updated_at = models.TextField(blank=True, null=True)
    promotionable = models.IntegerField(blank=True, null=True)
    meta_title = models.TextField(blank=True, null=True)
    discontinue_on = models.TextField(blank=True, null=True)
    vendor_id = models.IntegerField(blank=True, null=True)
    short_description = models.TextField(blank=True, null=True)
    how_to_use = models.TextField(blank=True, null=True)
    ingredients = models.TextField(blank=True, null=True)
    spree_brand_id = models.IntegerField(blank=True, null=True)
    spree_country_id = models.IntegerField(blank=True, null=True)
    why_we_love_it = models.TextField(blank=True, null=True)
    shipping_info = models.TextField(blank=True, null=True)
    good_to_know = models.TextField(blank=True, null=True)
    how_to_store = models.TextField(blank=True, null=True)
    avg_rating = models.FloatField(blank=True, null=True)
    reviews_count = models.IntegerField(blank=True, null=True)
    is_gift_card = models.IntegerField(blank=True, null=True)
    is_e_gift_card = models.IntegerField(blank=True, null=True)
    prod_id = models.IntegerField(blank=True, null=True)
    product_category = models.CharField(max_length=22, blank=True, null=True)
    caretype = models.CharField(max_length=1, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'api_products'


class ApiAllProductUniq(models.Model):
    id = models.IntegerField(primary_key=True)
    product_id = models.IntegerField(blank=True, null=True)
    product_name = models.TextField(blank=True, null=True)
    product_category = models.TextField(blank=True, null=True)
    product_cat_detail = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    how_to_use = models.TextField(blank=True, null=True)
    ingredients = models.TextField(blank=True, null=True)
    why_we_love_it = models.TextField(blank=True, null=True)
    good_to_know = models.TextField(blank=True, null=True)
    slug = models.TextField(blank=True, null=True)
    preference = models.TextField(blank=True, null=True)
    env_lifestyle = models.TextField(blank=True, null=True)
    hair_info = models.TextField(blank=True, null=True)
    skin_info = models.TextField(blank=True, null=True)
    all_info = models.TextField(blank=True, null=True)
    brand_id = models.IntegerField(blank=True, null=True)
    brand_name = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'api_all_product_uniq'


class customerquizresponse(models.Model):
    spree_option_id = models.IntegerField(blank=True, null=True)
    spree_question_id = models.IntegerField(blank=True, null=True)
    question_text = models.TextField(null=True, blank=True)
    option_text = models.TextField(null=True, blank=True)
    important_ingredients = models.TextField(null=True, blank=True)
    sess_id = models.TextField(null=True, blank=True)
    slug = models.CharField(max_length=10, null=True, blank=True)


class ApiQuizquestionsNew(models.Model):
    id = models.IntegerField(primary_key=True)
    quiztype = models.CharField(max_length=1, blank=True, null=True)
    question = models.TextField(blank=True, null=True)
    question_code = models.IntegerField(blank=True, null=True)
    choices = models.TextField(blank=True, null=True)
    choice_code = models.CharField(max_length=3, blank=True, null=True)
    recommendation = models.TextField(blank=True, null=True)
    non_recommendation = models.TextField(blank=True, null=True)
    tags = models.CharField(max_length=3, blank=True, null=True)
    taxons = models.CharField(max_length=39, blank=True, null=True)
    question_slug = models.CharField(max_length=4, blank=True, null=True)
    important_ingreds = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'api_quizquestions_new'


class ApiIngredients(models.Model):
    id = models.IntegerField(primary_key=True)
    ingredient = models.TextField(blank=True, null=True)
    url = models.TextField(blank=True, null=True)
    function_ingredients = models.TextField(blank=True, null=True)
    subtitle = models.TextField(blank=True, null=True)
    synonyms = models.TextField(blank=True, null=True)
    # Field name made lowercase.
    num_cas = models.TextField(db_column='num_CAS', blank=True, null=True)
    # Field name made lowercase.
    num_ec = models.TextField(db_column='num_EC', blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    detail = models.TextField(blank=True, null=True)
    # Field name made lowercase.
    chemical_iupac_name = models.TextField(
        db_column='Chemical_IUPAC_Name', blank=True, null=True)
    # Field name made lowercase.
    cosmetic_restrictions = models.TextField(
        db_column='Cosmetic_Restrictions', blank=True, null=True)
    # Field name made lowercase.
    ph_eur_name = models.TextField(
        db_column='Ph_Eur_Name', blank=True, null=True)
    irritancy = models.TextField(blank=True, null=True)
    # Field name made lowercase.
    comedogenicity = models.TextField(
        db_column='Comedogenicity', blank=True, null=True)
    uniq_ingredient = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'api_ingredients'


class ApiAllProductUniq(models.Model):
    id = models.IntegerField(primary_key=True)
    product_id = models.IntegerField(blank=True, null=True)
    product_name = models.TextField(blank=True, null=True)
    product_category = models.TextField(blank=True, null=True)
    product_cat_detail = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    how_to_use = models.TextField(blank=True, null=True)
    ingredients = models.TextField(blank=True, null=True)
    why_we_love_it = models.TextField(blank=True, null=True)
    good_to_know = models.TextField(blank=True, null=True)
    slug = models.TextField(blank=True, null=True)
    preference = models.TextField(blank=True, null=True)
    env_lifestyle = models.TextField(blank=True, null=True)
    hair_info = models.TextField(blank=True, null=True)
    skin_info = models.TextField(blank=True, null=True)
    all_info = models.TextField(blank=True, null=True)
    brand_id = models.IntegerField(blank=True, null=True)
    brand_name = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'api_all_product_uniq'


class SpreeTaxons(models.Model):
    id = models.IntegerField(primary_key=True)
    parent_id = models.IntegerField(blank=True, null=True)
    name = models.TextField(blank=True, null=True)
    permalink = models.TextField(blank=True, null=True)
    taxonomy_id = models.IntegerField(blank=True, null=True)
    depth = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'spree_taxons'


class ApiProductvectorNew(models.Model):
    id = models.IntegerField(primary_key=True)
    prod_name = models.TextField(blank=True, null=True)
    caretype = models.TextField(blank=True, null=True)
    product_category = models.TextField(blank=True, null=True)
    solvent = models.FloatField(blank=True, null=True)
    sunscreen = models.FloatField(blank=True, null=True)
    soothing = models.FloatField(blank=True, null=True)
    cell_communicating_ingredient = models.FloatField(blank=True, null=True)
    perfuming = models.FloatField(blank=True, null=True)
    emollient = models.FloatField(blank=True, null=True)
    emulsion_stabilising = models.FloatField(blank=True, null=True)
    moisturizer_humectant = models.FloatField(blank=True, null=True)
    buffering = models.FloatField(blank=True, null=True)
    skin_identical_ingredient = models.FloatField(blank=True, null=True)
    skin_brightening = models.FloatField(blank=True, null=True)
    viscosity_controlling = models.FloatField(blank=True, null=True)
    absorbent_mattifier = models.FloatField(blank=True, null=True)
    deodorant = models.FloatField(blank=True, null=True)
    antimicrobial_antibacterial = models.FloatField(blank=True, null=True)
    astringent = models.FloatField(blank=True, null=True)
    antioxidant = models.FloatField(blank=True, null=True)
    surfactant_cleansing = models.FloatField(blank=True, null=True)
    abrasive_scrub = models.FloatField(blank=True, null=True)
    colorant = models.FloatField(blank=True, null=True)
    emulsifying = models.FloatField(blank=True, null=True)
    anti_acne = models.FloatField(blank=True, null=True)
    preservative = models.FloatField(blank=True, null=True)
    exfoliant = models.FloatField(blank=True, null=True)
    chelating = models.FloatField(blank=True, null=True)
    prod_id = models.IntegerField(blank=True, null=True)
    env_lifestyle = models.TextField(blank=True, null=True)
    hair_info = models.TextField(blank=True, null=True)
    preference = models.TextField(blank=True, null=True)
    skin_info = models.TextField(blank=True, null=True)
    all_info = models.TextField(blank=True, null=True)
    brand = models.IntegerField(blank=True, null=True)
    key_ingredient = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'api_productvector_new'


class foundation_product_options(models.Model):
    product_id = models.IntegerField(null=True)
    product_name = models.TextField(null=True)
    hex_code = models.CharField(max_length=20, null=True)
    shades = models.CharField(max_length=20, null=True)
    phototype = models.CharField(max_length=50, null=True)


def user_directory_path(instance, filename):
    return 'logo/{0}'.format(filename)


class foundation_image(models.Model):
    imageid = models.CharField(max_length=200, null=True)
    imagename = models.ImageField(
        upload_to=user_directory_path,  blank=True, null=True)


#  Trryons Models

class tryons_image(models.Model):
    input_image = models.ImageField(
        upload_to=user_directory_path,  blank=True, null=True)
    output_image = models.ImageField(
        upload_to=user_directory_path,  blank=True, null=True)
    rgb_value = models.CharField(max_length=20, blank=True, null=True)
    sess_id = models.IntegerField(null=True)

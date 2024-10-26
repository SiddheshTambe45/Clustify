from django.shortcuts import render, redirect
from .forms import JSONUploadForm
import json
from .models import JSONData
# Create your views here.
from django.http import HttpResponse
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from datetime import datetime
from django.views.generic import View
# dataupload/views.py
from django.http import HttpResponse
from .models import JSONData, NumericData
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import JSONData, NumericData, Image
import json
# def upload_json(request):
#     if request.method == "POST":
#         form = JSONUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             file = form.cleaned_data['file']
#             json_data = json.load(file)
#
#             # Handle list or single dictionary structure in JSON
#             if isinstance(json_data, list):
#                 i = 1
#                 for item in json_data:
#                     JSONData.objects.create(
#                         name=item.get('name', 'Unknown'),
#                         age=item.get('age', 0),
#
#                     )
#                     print(i)
#                     i+=1
#             elif isinstance(json_data, dict):
#                 JSONData.objects.create(
#                     name=json_data.get('name', 'Unknown'),
#                     age=json_data.get('age', 0),
#
#                 )
#
#             return redirect('success')
#     else:
#         form = JSONUploadForm()
#     return render(request, 'upload.html', {'form': form})

import os
from django.conf import settings
from django.shortcuts import render

def analysis_view(request):
    # Query images from Image model
    # images = Image.objects.all()  # Adjust to show only 2 images if desired
    #
    # # Render images to the template
    # return render(request, 'analysis.html', {'imagess': images})
    plots_dir = os.path.join(settings.PLOTS_ROOT)  # Make sure PLOTS_ROOT is set in your settings.py

    # Get a list of all image files in the plots directory
    image_files = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Create full URLs for the images
    images = [os.path.join(settings.PLOTS_URL, f) for f in image_files]

    # Render images to the template
    return render(request, 'plot_analysis.html', {'images': images})

def upload_json(request):
    if request.method == "POST":
        form = JSONUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['file']
            json_data = json.load(file)

            # Handle list or single dictionary structure in JSON
            if isinstance(json_data, list):
                for item in json_data:
                    JSONData.objects.create(
                        CustomerID=item.get('CustomerID', 'unknown'),
                        Name=item.get('Name', 'unknown'),
                        Age=item.get('Age', None),  # Allow None if not provided
                        Gender=item.get('Gender', 'Not specified'),
                        Region=item.get('Region', 'Not specified'),
                        City=item.get('City', 'Not specified'),
                        Interests=item.get('Interests', ''),
                        Purchase_History=item.get('Purchase History', {}),  # Assuming this is a dictionary
                        Last_Purchase_Date=item.get('Last Purchase Date', None),
                        Feedback=item.get('Feedback', ''),
                        Income_Level=item.get('Income Level', 'Not specified')
                    )
            elif isinstance(json_data, dict):
                JSONData.objects.create(
                    CustomerID=json_data.get('CustomerID', 'unknown'),
                    Name=json_data.get('Name', 'unknown'),
                    Age=json_data.get('Age', None),
                    Gender=json_data.get('Gender', 'Not specified'),
                    Region=json_data.get('Region', 'Not specified'),
                    City=json_data.get('City', 'Not specified'),
                    Interests=json_data.get('Interests', ''),
                    Purchase_History=json_data.get('Purchase History', {}),
                    Last_Purchase_Date=json_data.get('Last Purchase Date', None),
                    Feedback=json_data.get('Feedback', ''),
                    Income_Level=json_data.get('Income Level', 'Not specified')
                )

            return redirect('records')  # Ensure 'success' is a valid URL name in your urlpatterns
    else:
        form = JSONUploadForm()
    return render(request, 'upload.html', {'form': form})


def success(request):
    return HttpResponse("File uploaded successfully!")


def records_view(request):
    records = JSONData.objects.all()  # Fetch all records from the database
    return render(request, 'records.html', {'records': records})

def transform_data(request):
    # # Get all records from JSONData
    # json_records = JSONData.objects.all()
    #
    # # Create mapping dictionaries for categorical values
    # gender_mapping = {'Male': 1, 'Female': 2, 'Other': 3, 'Not specified': 0}
    #
    # # Create dynamic mappings for region, city, and income level
    # region_mapping = defaultdict(lambda: len(region_mapping) + 1)
    # region_mapping['Not specified'] = 0
    #
    # city_mapping = defaultdict(lambda: len(city_mapping) + 1)
    # city_mapping['Not specified'] = 0
    #
    # income_mapping = {
    #     'Low': 1,
    #     'Medium': 2,
    #     'High': 3,
    #     'Not specified': 0
    # }
    #
    # def process_purchase_history(purchase_history):
    #     if not purchase_history:
    #         return {
    #             'total_purchases': 0,
    #             'average_amount': 0,
    #             'purchase_frequency': 0,
    #             'category_counts': {},
    #             'monetary_value': 0
    #         }
    #
    #     try:
    #         # Initialize metrics
    #         total_purchases = len(purchase_history)
    #         total_amount = sum(float(purchase.get('amount', 0)) for purchase in purchase_history)
    #         average_amount = total_amount / total_purchases if total_purchases > 0 else 0
    #
    #         # Calculate purchase frequency (purchases per month)
    #         if total_purchases >= 2:
    #             dates = [datetime.strptime(purchase.get('date', ''), '%Y-%m-%d')
    #                      for purchase in purchase_history
    #                      if purchase.get('date')]
    #             if dates:
    #                 date_range = (max(dates) - min(dates)).days / 30  # Convert to months
    #                 purchase_frequency = total_purchases / date_range if date_range > 0 else 0
    #             else:
    #                 purchase_frequency = 0
    #         else:
    #             purchase_frequency = 0
    #
    #         # Count purchases by category
    #         category_counts = {}
    #         for purchase in purchase_history:
    #             category = purchase.get('category', 'unknown')
    #             category_counts[category] = category_counts.get(category, 0) + 1
    #
    #         return {
    #             'total_purchases': total_purchases,
    #             'average_amount': round(average_amount, 2),
    #             'purchase_frequency': round(purchase_frequency, 2),
    #             'category_counts': category_counts,
    #             'monetary_value': round(total_amount, 2)
    #         }
    #     except Exception as e:
    #         print(f"Error processing purchase history: {e}")
    #         return {
    #             'total_purchases': 0,
    #             'average_amount': 0,
    #             'purchase_frequency': 0,
    #             'category_counts': {},
    #             'monetary_value': 0
    #         }
    # # Function to convert feedback text to numeric value
    # def calculate_feedback_score(feedback_text):
    #     if not feedback_text:
    #         return 0
    #
    #     # Simple sentiment analysis (you can make this more sophisticated)
    #     positive_words = ['good', 'great', 'excellent', 'amazing', 'happy', 'satisfied']
    #     negative_words = ['bad', 'poor', 'terrible', 'unhappy', 'disappointed']
    #
    #     feedback_lower = feedback_text.lower()
    #     positive_count = sum(1 for word in positive_words if word in feedback_lower)
    #     negative_count = sum(1 for word in negative_words if word in feedback_lower)
    #
    #     return positive_count - negative_count
    #
    # # Convert interests to numeric representation
    # def process_interests(interests_text):
    #     if not interests_text:
    #         return {}
    #
    #     # Split interests by comma and create a binary representation
    #     interests_list = [interest.strip() for interest in interests_text.split(',')]
    #     return {interest: 1 for interest in interests_list}
    #
    # # Process each record
    # for json_record in json_records:
    #     # Check if record already exists
    #     numeric_record, created = NumericData.objects.get_or_create(
    #         CustomerID=json_record.CustomerID
    #     )
    #
    #     # Update numeric values
    #     numeric_record.Age = json_record.Age if json_record.Age is not None else 0
    #     numeric_record.Gender = gender_mapping.get(json_record.Gender, 0)
    #     numeric_record.Region = region_mapping[json_record.Region]
    #     numeric_record.City = city_mapping[json_record.City]
    #     numeric_record.Interests = process_interests(json_record.Interests)
    #     numeric_record.Feedback = calculate_feedback_score(json_record.Feedback)
    #     numeric_record.Income_Level = income_mapping.get(json_record.Income_Level, 0)
    #     numeric_record.Purchase_History = process_purchase_history(json_record.Purchase_History)
    #
    #     # Save the record
    #     numeric_record.save()
    #
    # # Create a summary of the mappings
    # mappings_summary = {
    #     'gender_mapping': gender_mapping,
    #     'region_mapping': dict(region_mapping),
    #     'city_mapping': dict(city_mapping),
    #     'income_mapping': income_mapping
    # }
    #
    # num_obj = NumericData.objects.all()
    # print(num_obj)
    # # Return the mappings for reference
    # return render(request, 'transformation_summary.html', {
    #     'mappings': mappings_summary,
    #     'records_processed': len(json_records),
    #     'records':num_obj
    #
    # })

    # Get all records from JSONData
    # Get all records from JSONData
    json_records = JSONData.objects.all()

    # Define fixed mappings based on the provided documentation
    FEEDBACK_MAPPING = {
        'Very satisfied': 4,
        'Satisfied': 3,
        'Neutral': 2,
        'Dissatisfied': 1,
        'Not specified': 0
    }

    REGION_MAPPING = {
        'North': 1,
        'West': 2,
        'South': 3,
        'Central': 4,
        'East': 5,
        'Not specified': 0
    }

    INCOME_MAPPING = {
        'Lower': 1,
        'Middle': 2,
        'Upper': 3,
        'Not specified': 0
    }

    GENDER_MAPPING = {
        'Female': 1,
        'Male': 2,
        'Not specified': 0
    }

    CITY_MAPPING = {
        'Delhi': 1, 'Mumbai': 2, 'Bengaluru': 3, 'Lucknow': 4,
        'Kolkata': 5, 'Chennai': 6, 'Hyderabad': 7, 'Pune': 8,
        'Ahmedabad': 9, 'Jaipur': 10, 'Noida': 11, 'Bhubaneswar': 12,
        'Surat': 13, 'Kochi': 14, 'Nagpur': 15, 'Agra': 16, 'Indore': 17,
        'Not specified': 0
    }

    INTERESTS_MAPPING = {
        'Shopping': 1, 'Travel': 2, 'Technology': 3, 'Sports': 4,
        'Fashion': 5, 'Music': 6, 'Fitness': 7, 'Cooking': 8,
        'Photography': 9, 'Gaming': 10, 'Accessories': 11, 'Gadgets': 12,
        'Kitchen Utensils': 13, 'Clothing': 14, 'Gym Equipment': 15,
        'Travel Gear': 16, 'Yoga Accessories': 17, 'Smartwatch': 18,
        'Fitness Gear': 19, 'Music Instruments': 20, 'Cooking Books': 21,
        'Camera': 22, 'Cookbook': 23, 'Fitness Classes': 24,
        'Cooking Utensils': 25
    }

    PURCHASE_MAPPING = {
        'Clothing': 1, 'Electronics': 2, 'Cooking Utensils': 3, 'Gadgets': 4,
        'Fitness Equipment': 5, 'Cameras': 6, 'Gaming Console': 7, 'Gym Equipment': 8,
        'Cooking Books': 9, 'Travel Guide': 10, 'Yoga Mat': 11, 'Smartwatch': 12,
        'Cookbook': 13, 'Smartphone': 14, 'Travel Backpack': 15,
        'Technology': 16, 'Kitchen Utensils': 17, 'Fashion Accessories': 18,'Fitness Classes':19,'Fitness Gear':20
    }

    def parse_array_field(field_value):
        """Parse array field that might be stored as string"""
        if isinstance(field_value, str):
            try:
                # Try to parse if it's a JSON string
                return json.loads(field_value)
            except json.JSONDecodeError:
                # If it's a comma-separated string
                return [x.strip() for x in field_value.split(',') if x.strip()]
        elif isinstance(field_value, list):
            return field_value
        elif field_value is None:
            return []
        else:
            return [field_value]

    def convert_interests(interests_data):
        """Convert interests to numeric values"""
        # First parse the interests data
        interests = parse_array_field(interests_data)

        # If interests are already numbers, return as is
        if interests and all(isinstance(x, (int, float)) for x in interests):
            return interests

        # Convert to numeric values
        numeric_interests = []
        for interest in interests:
            if isinstance(interest, (int, float)):
                numeric_interests.append(int(interest))
            else:
                numeric_value = INTERESTS_MAPPING.get(str(interest).strip(), 0)
                if numeric_value != 0:
                    numeric_interests.append(numeric_value)

        return numeric_interests

    def convert_purchase_history(purchase_data):
        """Convert purchase history to numeric values"""
        # First parse the purchase history data
        purchases = parse_array_field(purchase_data)

        # If purchases are already numbers, return as is
        if purchases and all(isinstance(x, (int, float)) for x in purchases):
            return purchases

        # Convert to numeric values
        numeric_purchases = []
        for purchase in purchases:
            if isinstance(purchase, (int, float)):
                numeric_purchases.append(int(purchase))
            else:
                numeric_value = PURCHASE_MAPPING.get(str(purchase).strip(), 0)
                if numeric_value != 0:
                    numeric_purchases.append(numeric_value)

        return numeric_purchases

    def process_date(date_str):
        """Convert date string to datetime object"""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            return None

    # Process each record
    transformed_records = []
    errors = []

    for record in json_records:
        try:
            numeric_record, created = NumericData.objects.get_or_create(
                CustomerID=record.CustomerID
            )

            # Basic fields
            numeric_record.Age = record.Age if record.Age is not None else 0
            numeric_record.Gender = GENDER_MAPPING.get(record.Gender, 0)
            numeric_record.Region = REGION_MAPPING.get(record.Region, 0)
            numeric_record.City = CITY_MAPPING.get(record.City, 0)
            numeric_record.Feedback = FEEDBACK_MAPPING.get(record.Feedback, 0)
            numeric_record.Income_Level = INCOME_MAPPING.get(record.Income_Level, 0)

            # Convert and store Interests
            interests = convert_interests(record.Interests)
            numeric_record.Interests = interests

            # Convert and store Purchase History
            purchases = convert_purchase_history(record.Purchase_History)
            numeric_record.Purchase_History = purchases

            # Process date
            numeric_record.Last_Purchase_Date = process_date(record.Last_Purchase_Date)

            # Additional metrics
            numeric_record.Total_Purchases = len(purchases)
            numeric_record.Unique_Categories = len(set(purchases))

            # Debug logging
            print(f"Record {record.CustomerID}:")
            print(f"Original Interests: {record.Interests}")
            print(f"Converted Interests: {interests}")
            print(f"Original Purchase History: {record.Purchase_History}")
            print(f"Converted Purchase History: {purchases}")

            numeric_record.save()
            transformed_records.append(numeric_record)

        except Exception as e:
            error_msg = f"Error processing record {record.CustomerID}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
            continue

    # Create summary statistics
    summary = {
        'total_records': len(json_records),
        'successfully_transformed': len(transformed_records),
        'errors': len(errors),
        'error_messages': errors,
        'mappings': {
            'feedback': FEEDBACK_MAPPING,
            'region': REGION_MAPPING,
            'income': INCOME_MAPPING,
            'gender': GENDER_MAPPING,
            'city': CITY_MAPPING,
            'interests': INTERESTS_MAPPING,
            'purchases': PURCHASE_MAPPING
        }
    }
    num_obj = NumericData.objects.all()
    return render(request, 'transformation_summary.html', {
        'summary': summary,
        'records': transformed_records,
        'records': num_obj
    })


import requests
from django.http import JsonResponse
import os
from dotenv import load_dotenv
from .models import ClusterAnalysis

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_SECRET = os.getenv('GEMINI_API_SECRET')

conversation_history = []


def analyze_cluster_with_chat(request):
    if request.method == 'POST':
        user_input = request.POST.get('message')

        # Retrieve data from the ClusterAnalysis model
        analyses = ClusterAnalysis.objects.all().values()
        data_to_send = list(analyses)  # Convert QuerySet to list

        # Prepare the conversation for Gemini
        messages = []

        if user_input:
            # Add user message to the conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Create a prompt with the cluster analysis data and the user message
            prompt = (
                    "Here is some cluster analysis data:\n"
                    f"{data_to_send}\n\n"
                    "Based on this data, " + user_input
            )

            # Sending the prompt to Gemini LLM
            response = requests.post(
                "https://api.gemini.example.com/chat",  # Replace with actual chat endpoint
                headers={
                    'Authorization': f'Bearer {GEMINI_API_KEY}',
                    'Content-Type': 'application/json',
                },
                json={"prompt": prompt}
            )

            # Check the response
            if response.status_code == 200:
                gemini_response = response.json()
                model_reply = gemini_response.get('choices', [{}])[0].get('message', {}).get('content', '')

                # Append model's reply to conversation history
                conversation_history.append({"role": "assistant", "content": model_reply})

                return JsonResponse({"reply": model_reply}, status=200)

            return JsonResponse({"error": "Failed to get a response from Gemini"}, status=response.status_code)

    return JsonResponse({"error": "Invalid request method"}, status=400)

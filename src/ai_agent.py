# Import the required libraries and methods
import requests
from typing import List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import ollama

def processing_and_extracting_infromation_from_image(image_path, user_query):
    """Extract the information from the images realted to user queries"""
    response = ollama.chat(
        model='llama3.2-vision:latest',
        messages=[
            {
                'role': 'Assistant',
                'content': 'Extract information from images related to objects, shape and node details like number os shapes with position or node details. and try to take the help from the user quesry to get the idea about the user question and what information they actually want. user question - {user_query}',
                'images': ['image_path']
            }
        ]
    )
    return response.message.content

def immage_object_shapes_correlation(responses, user_query):
    """Find the correlation between the images uploaded by the user and return the correlationa dn similarities."""
    response = ollama.chat(
        model='llama3.2-vision:latest',
        messages=[
            {
                'role': 'Assistant',
                'content': 'Images information {response},Based on this information provided for the images, find the correlation between the images and return the all information,  try to take the help from the user quesry to get the idea about the user question and what information they actually want. user question - {user_query} ',
            }
        ]
    )
    return response.message.content

def response_generation():
    
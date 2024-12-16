import ollama

response = ollama.chat(
    model='llama3.2-vision:latest',
    messages=[
        {
            'role': 'user',
            'content': 'Take the two images. Correlate them and answer this simple question. What are the nodes of the orange colored polygon or other polygons ? using any VLLMs',
            'images': ['images\image_1.jpg']
        }
    ]
)

print(response)
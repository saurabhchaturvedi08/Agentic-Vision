from langchain.agents import initialize_agent, Tool
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from src.node_detection import detect_polygons
from src.polygon_correlation import correlate_polygons
from src.generate_response import generate_response
import torch

model_id = "meta-llama/Llama-3.2-1B"

# Create the HuggingFace pipeline
llama_pipeline = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Define tools
tools = [
    Tool(
        name="DetectPolygons",
        func=detect_polygons,
        description="Detect polygons in an image and return their nodes."
    ),
    Tool(
        name="CorrelatePolygons",
        func=correlate_polygons,
        description="Correlate polygons between two images and calculate similarity."
    ),
    Tool(
        name="GenerateResponse",
        func=generate_response,
        description="Generate a human-readable response summarizing the findings."
    )
]

# Initialize agent
llm = HuggingFacePipeline(pipeline=llama_pipeline)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Example usage
combined_query = (
    "What are the nodes of the orange polygon in both images? "
    "Image 1 is located at 'images/image_1.jpg' and Image 2 is located at 'images/image_2.jpg'."
)
response = agent.invoke({"input": combined_query})
print(response)

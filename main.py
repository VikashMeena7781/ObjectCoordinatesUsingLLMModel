import os 

os.environ['OPENAI_API_KEY'] = 'your api key'

from openai import OpenAI
from dotenv import load_dotenv
from utils import draw_circle, encode_image
import json

load_dotenv()

client = OpenAI()


def ask_gpt4_vision(prompt, object_to_detect, image_path):
    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            max_tokens=100,
            messages=[
                {
                    "role": "system", 
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Detect: {object_to_detect}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.45,
            top_p=0,
        )
        # print("Response ",response)
        content = response.choices[0].message.content
        # print("content...",content)
        json_str = content.strip('`json\n') # Extract the JSON part from the string (remove the ```json and ``` at both ends)
        # print("json_str ",json_str)
        coordinates = json.loads(json_str) # Convert the JSON string into a Python dictionary
        # print(coordinates)
        

        print('-' * 30)
        print("Detect:", object_to_detect)
        print("Details:", coordinates["details"])
        print(f"Coordinates: [{coordinates['x']}, {coordinates['y']}]")
        print('-' * 30)

    except Exception as e:
        print(e)
        coordinates = {"x": 0, "y": 0, "details": ""}
    
    return coordinates

image_path = "assets/laptop_pc.jpeg"
# image_path = "assets/puppy.jpg"

prompt = """
As an image recognition expert, your task is to analyze images and provide 
output in JSON format with the following keys only: 'x', 'y', and 'details'.

- 'x' and 'y' should represent the coordinates of the center of the detected 
object within the image, with the reference point [0,0] at the top left corner.
- 'details' should provide a brief description of the object identified in the image.

For cases involving the identification of people or animals, focus on locating and 
identifying the face of the person or animal. Ensure that the given 'x' and 'y' 
coordinates correspond to the center of the identified face.

Please adhere strictly to this output structure:
{
  "x": value,
  "y": value,
  "details": "Description"
}

Note: Do not include any additional data or keys outside of what has been specified.
"""

query = "detect laptop"

coordinates = ask_gpt4_vision(prompt, query, image_path)
draw_circle(image_path, coordinates)

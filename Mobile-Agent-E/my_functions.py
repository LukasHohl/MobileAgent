import base64
from openai import OpenAI
from os import environ
#code is mostly from the openai documentation
#oh and it is obviously based on the code from process_image
#https://platform.openai.com/docs/guides/vision
def process_image2(image, query, caption_model=None):
    client = OpenAI(api_key=environ["OPENAI_API_KEY"])
    with open(image, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    response = client.chat.completions.create(
    model="gpt-4o-mini", #TODO find out best settings? log usage data
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)
    try:
        response = response.choices[0].message.content
    except:
        response = "This is an icon."
    
    return response
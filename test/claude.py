import base64

from anthropic import Anthropic

client = Anthropic()
MODEL_NAME = "claude-opus-4-1"

with open("../images/sunset.jpeg", "rb") as image_file:
    binary_data = image_file.read()
    base_64_encoded_data = base64.b64encode(binary_data)
    base64_string = base_64_encoded_data.decode("utf-8")


message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_string,
                },
            },
            {"type": "text", "text": "Write a sonnet based on this image."},
        ],
    }
]

response = client.messages.create(
    model=MODEL_NAME, max_tokens=2048, messages=message_list
)
print(response.content[0].text)

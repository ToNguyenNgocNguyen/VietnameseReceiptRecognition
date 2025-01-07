import requests

def inference_ocr(image, url, headers, prompt):
    """
    Send an image to the server for inference and return the response.

    Args:
    - image (str): The image file.
    - url (str): The API endpoint URL.
    - headers (dict): Request headers.
    - prompt (str): The prompt to be used for inference.

    Returns:
    - dict: JSON response from the server if successful, otherwise error message.
    """

    # Form data
    files = {
        'file': image
    }
    data = {
        'prompt': prompt
    }

    # Make the POST request
    response = requests.post(url, headers=headers, files=files, data=data)

    # Check the response
    if response.status_code == 200:
        return response.json()  # Return the JSON response
    else:
        return {'error': f"Error {response.status_code}: {response.text}"}  # Return error details


if __name__=="__main__":
    url = 'https://b39f-34-19-100-163.ngrok-free.app/api/inference'
    headers = {
        'accept': 'application/json',
    }
    image_path = '/home/lavender/My-Project/CV_Project/05_receipt_recognition/image.png'
    prompt = 'Hãy đọc các văn bản có trong ảnh.'

    response = inference_ocr(image_path, url, headers, prompt)
    print(response)

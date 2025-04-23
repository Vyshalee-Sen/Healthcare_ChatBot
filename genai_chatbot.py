from google import genai

client = genai.Client(api_key="AIzaSyAewX2hP6b5j7I7G657_tqEzFEnaXyv1Rc")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="what are the symptoms of cancer",
)

print(response.text)
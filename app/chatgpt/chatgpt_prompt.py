import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = "your-api-key"


def generate_response(prompt, max_tokens=150, temperature=0.7, top_p=1.0):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    prompt = "Explain the concept of machine learning in simple terms."
    response = generate_response(prompt)
    print(response)

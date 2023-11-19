from gradio_client import Client
import numpy as np
import gradio as gr
import requests
import json
import dotenv
import soundfile as sf
import time
import textwrap
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os
import uuid


welcome_message = """
# 👋🏻Welcome to ⚕🗣️😷MultiMed - Access Chat ⚕🗣️😷

🗣️📝 This is an educational and accessible conversational tool.

### How To Use ⚕🗣️😷MultiMed⚕: 

🗣️📝Interact with ⚕🗣️😷MultiMed⚕ in any language using image, audio or text!

📚🌟💼 that uses [Tonic/stablemed](https://huggingface.co/Tonic/stablemed) and [adept/fuyu-8B](https://huggingface.co/adept/fuyu-8b) with [Vectara](https://huggingface.co/vectara) embeddings + retrieval. 
do [get in touch](https://discord.gg/GWpVpekp). You can also use 😷MultiMed⚕️ on your own data & in your own way by cloning this space. 🧬🔬🔍 Simply click here: <a style="display:inline-block" href="https://huggingface.co/spaces/TeamTonic/MultiMed?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></h3>
### Join us : 

🌟TeamTonic🌟 is always making cool demos! Join our active builder's🛠️community on 👻Discord: [Discord](https://discord.gg/GWpVpekp) On 🤗Huggingface: [TeamTonic](https://huggingface.co/TeamTonic) & [MultiTransformer](https://huggingface.co/MultiTransformer) On 🌐Github: [Polytonic](https://github.com/tonic-ai) & contribute to 🌟 [PolyGPT](https://github.com/tonic-ai/polygpt-alpha)"             
"""


languages = {
    "English": "eng",
    "Modern Standard Arabic": "arb",
    "Bengali": "ben",
    "Catalan": "cat",
    "Czech": "ces",
    "Mandarin Chinese": "cmn",
    "Welsh": "cym",
    "Danish": "dan",
    "German": "deu",
    "Estonian": "est",
    "Finnish": "fin",
    "French": "fra",
    "Hindi": "hin",
    "Indonesian": "ind",
    "Italian": "ita",
    "Japanese": "jpn",
    "Korean": "kor",
    "Maltese": "mlt",
    "Dutch": "nld",
    "Western Persian": "pes",
    "Polish": "pol",
    "Portuguese": "por",
    "Romanian": "ron",
    "Russian": "rus",
    "Slovak": "slk",
    "Spanish": "spa",
    "Swedish": "swe",
    "Swahili": "swh",
    "Telugu": "tel",
    "Tagalog": "tgl",
    "Thai": "tha",
    "Turkish": "tur",
    "Ukrainian": "ukr",
    "Urdu": "urd",
    "Northern Uzbek": "uzn",
    "Vietnamese": "vie"
}


# Global variables to hold component references
components = {}
dotenv.load_dotenv()
seamless_client = Client("facebook/seamless_m4t")
HuggingFace_Token = os.getenv("HuggingFace_Token")
hf_token = os.getenv("HuggingFace_Token")
base_model_id = os.getenv('BASE_MODEL_ID', 'default_base_model_id')
model_directory = os.getenv('MODEL_DIRECTORY', 'default_model_directory')
device = "cuda" if torch.cuda.is_available() else "cpu"

image_description = "" 
# audio_output = ""
# global markdown_output
# global audio_output


def check_hallucination(assertion, citation):
    api_url = "https://api-inference.huggingface.co/models/vectara/hallucination_evaluation_model"
    header = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": f"{assertion} [SEP] {citation}"}

    response = requests.post(api_url, headers=header, json=payload, timeout=120)
    output = response.json()
    output = output[0][0]["score"]

    return f"**hallucination score:** {output}"


# Define the API parameters
vapi_url = "https://api-inference.huggingface.co/models/vectara/hallucination_evaluation_model"

headers = {"Authorization": f"Bearer {hf_token}"}


# Function to query the API
def query(payload):
    response = requests.post(vapi_url, headers=headers, json=payload)
    return response.json()


# Function to evaluate hallucination
def evaluate_hallucination(input1, input2):
    # Combine the inputs
    combined_input = f"{input1}. {input2}"
    
    # Make the API call
    output = query({"inputs": combined_input})
    
    # Extract the score from the output
    score = output[0][0]['score']
    
    # Generate a label based on the score
    if score < 0.5:
        label = f"🔴 High risk. Score: {score:.2f}"
    else:
        label = f"🟢 Low risk. Score: {score:.2f}"
    
    return label


def save_audio(audio_input, output_dir="saved_audio"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract sample rate and audio data
    sample_rate, audio_data = audio_input

    # Generate a unique file name
    file_name = f"audio_{int(time.time())}.wav"
    file_path = os.path.join(output_dir, file_name)

    # Save the audio file
    sf.write(file_path, audio_data, sample_rate)

    return file_path


def save_image(image_input, output_dir="saved_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Assuming image_input is a NumPy array
    if isinstance(image_input, np.ndarray):
        # Convert NumPy arrays to PIL Image
        image = Image.fromarray(image_input)

        # Generate a unique file name
        file_name = f"image_{int(time.time())}.png"
        file_path = os.path.join(output_dir, file_name)

        # Save the image file
        image.save(file_path)

        return file_path
    else:
        raise ValueError("Invalid image input type")


def process_speech(input_language, audio_input):
    """
    processing sound using seamless_m4t
    """
    if audio_input is None:
        return "no audio or audio did not save yet \nplease try again ! "
    print(f"audio : {audio_input}")
    print(f"audio type : {type(audio_input)}")
    out = seamless_client.predict(
        "S2TT",
        "file",
        None,
        audio_input,
        "",
        input_language,
        "English",
        api_name="/run",
    )
    out = out[1]  # get the text
    try:
        return f"{out}"
    except Exception as e:
        return f"{e}"


def is_base64(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

def convert_text_to_speech(input_text: str, source_language: str, target_language: str) -> tuple[str, str]:
    client = Client("https://facebook-seamless-m4t.hf.space/--replicas/8cllp/")

    try:
        # Make a prediction request to the client
        result = client.predict(
            "T2ST",
            "text",  # Since we are doing text-to-speech
            None,
            None,
            input_text,
            source_language,
            target_language,
            api_name="/run"
        )

        # Print or log the raw API response for inspection
        print("Raw API Response:", result)

        # Initialize variables
        translated_text = ""
        audio_file_path = ""

        # Process the result
        if result:
            for item in result:
                if isinstance(item, str):
                    # Check if the item is a URL pointing to an audio file or a base64 encoded string
                    if any(ext in item.lower() for ext in ['.mp3', '.wav', '.ogg']) or is_base64(item):
                        if not audio_file_path:  # Store only the first audio file path or base64 string
                            audio_file_path = item
                    else:
                        # Concatenate the translated text
                        translated_text += item + " "

        return audio_file_path, translated_text.strip()

    except Exception as e:
        print(f"Error in text-to-speech conversion: {str(e)}")
        return None, f"Error in text-to-speech conversion: {str(e)}"


def query_vectara(text):
    user_message = text

    # Read authentication parameters from the .env file
    customer_id = os.getenv('CUSTOMER_ID')
    corpus_id = os.getenv('CORPUS_ID')
    api_key = os.getenv('API_KEY')

    # Define the headers
    api_key_header = {
        "customer-id": customer_id,
        "x-api-key": api_key
    }

    # Define the request body in the structure provided in the example
    request_body = {
        "query": [
            {
                "query": user_message,
                "queryContext": "",
                "start": 1,
                "numResults": 25,
                "contextConfig": {
                    "charsBefore": 0,
                    "charsAfter": 0,
                    "sentencesBefore": 2,
                    "sentencesAfter": 2,
                    "startTag": "%START_SNIPPET%",
                    "endTag": "%END_SNIPPET%",
                },
                "rerankingConfig": {
                    "rerankerId": 272725718,
                    "mmrConfig": {
                        "diversityBias": 0.35
                    }
                },
                "corpusKey": [
                    {
                        "customerId": customer_id,
                        "corpusId": corpus_id,
                        "semantics": 0,
                        "metadataFilter": "",
                        "lexicalInterpolationConfig": {
                            "lambda": 0
                        },
                        "dim": []
                    }
                ],
                "summary": [
                    {
                        "maxSummarizedResults": 5,
                        "responseLang": "auto",
                        "summarizerPromptName": "vectara-summary-ext-v1.2.0"
                    }
                ]
            }
        ]
    }

    # Make the API request using Gradio
    response = requests.post(
        "https://api.vectara.io/v1/query",
        json=request_body,  # Use json to automatically serialize the request body
        verify=True,
        headers=api_key_header
    )

    if response.status_code == 200:
        query_data = response.json()
        if query_data:
            sources_info = []

            # Extract the summary.
            summary = query_data['responseSet'][0]['summary'][0]['text']

            # Iterate over all response sets
            for response_set in query_data.get('responseSet', []):
                # Extract sources
                # Limit to top 5 sources.
                for source in response_set.get('response', [])[:5]:
                    source_metadata = source.get('metadata', [])
                    source_info = {}

                    for metadata in source_metadata:
                        metadata_name = metadata.get('name', '')
                        metadata_value = metadata.get('value', '')

                        if metadata_name == 'title':
                            source_info['title'] = metadata_value
                        elif metadata_name == 'author':
                            source_info['author'] = metadata_value
                        elif metadata_name == 'pageNumber':
                            source_info['page number'] = metadata_value

                    if source_info:
                        sources_info.append(source_info)

            result = {"summary": summary, "sources": sources_info}
            return f"{json.dumps(result, indent=2)}"
        else:
            return "No data found in the response."
    else:
        return f"Error: {response.status_code}"


# Functions to Wrap the Prompt Correctly
def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


def multimodal_prompt(user_input, system_prompt="You are an expert medical analyst:"):

    # Combine user input and system prompt
    formatted_input = f"{user_input}{system_prompt}"

    # Encode the input text
    encodeds = tokenizer(formatted_input, return_tensors="pt", add_special_tokens=False)
    model_inputs = encodeds.to(device)

    # Generate a response using the model //MODEL UNDEFINED, using peft_model instead.
    output = peft_model.generate(
        **model_inputs,
        max_length=512,
        use_cache=True,
        early_stopping=True,
        bos_token_id=peft_model.config.bos_token_id,
        eos_token_id=peft_model.config.eos_token_id,
        pad_token_id=peft_model.config.eos_token_id,
        temperature=0.1,
        do_sample=True
    )

    # Decode the response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return response_text


# Instantiate the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t", token=hf_token, trust_remote_code=True, padding_side="left")
# tokenizer = AutoTokenizer.from_pretrained("Tonic/stablemed", trust_remote_code=True, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Load the PEFT model
peft_config = PeftConfig.from_pretrained("Tonic/stablemed", token=hf_token)
peft_model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t", token=hf_token, trust_remote_code=True)
peft_model = PeftModel.from_pretrained(peft_model, "Tonic/stablemed", token=hf_token)


class ChatBot:
    def __init__(self):
        self.history = []
        
    @staticmethod
    def doctor(user_input, system_prompt="You are an expert medical analyst:"):
        formatted_input = f"{system_prompt}{user_input}"
        user_input_ids = tokenizer.encode(formatted_input, return_tensors="pt")
        response = peft_model.generate(input_ids=user_input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        return response_text


bot = ChatBot()


def process_summary_with_stablemed(summary):
    system_prompt = "You are a medical instructor . Assess and describe the proper options to your students in minute detail. Propose a course of action for them to base their recommendations on based on your description."
    response_text = bot.doctor(summary, system_prompt)
    return response_text

    
# Main function to handle the Gradio interface logic

def process_and_query(input_language=None, audio_input=None, image_input=None, text_input=None):
    try:
        
        combined_text = ""
        markdown_output = ""  
        image_text = ""  
        language_code = None

        # Convert input language to its code
        if input_language and input_language in languages:
            language_code = languages[input_language]

        # Debugging print statement
        print(f"Image Input Type: {type(image_input)}, Audio Input Type: {type(audio_input)}")
        
        # Process image input
        if image_input is not None:
            # Convert image_input to a file path
            image_file_path = save_image(image_input)
            image_text = process_image(image_file_path)
            combined_text += "\n\n**Image Input:**\n" + image_text

        # Process audio input
        elif audio_input is not None:
            audio_file_path = save_audio(audio_input)
            audio_text = process_speech(input_language, audio_file_path)        
            combined_text += "\n\n**Audio Input:**\n" + audio_text

        # Process text input
        elif text_input is not None and text_input.strip():
            combined_text += "The user asks the following to his health adviser: " + text_input

        # Check if combined text is empty
        else:
            return "Error: Please provide some input (text, audio, or image)."

        # Append the original image description in Markdown
        if image_text:
            markdown_output += "\n### Original Image Description\n"
            markdown_output += image_text + "\n"
    
        # Use the text to query Vectara
        vectara_response_json = query_vectara(combined_text)

        # Parse the Vectara response
        vectara_response = json.loads(vectara_response_json)
        summary = vectara_response.get('summary', 'No summary available')
        sources_info = vectara_response.get('sources', [])

        # Format Vectara response in Markdown
        markdown_output = "### Vectara Response Summary\n"
        markdown_output += f"* **Summary**: {summary}\n"
        markdown_output += "### Sources Information\n"
        for source in sources_info:
            markdown_output += f"* {source}\n"

        # Process the summary with Stablemed
        final_response = process_summary_with_stablemed(summary)

        # Convert translated text to speech and get both audio file and text
        target_language = "English"  # Set the target language for the speech
        audio_output, translated_text = convert_text_to_speech(final_response, target_language, input_language)
        
        # Evaluate hallucination
        hallucination_label = evaluate_hallucination(final_response, summary)

        # Add final response and hallucination label to Markdown output
        markdown_output += "\n### Processed Summary with StableMed\n"
        markdown_output += final_response + "\n"
        markdown_output += "\n### Hallucination Evaluation\n"
        markdown_output += f"* **Label**: {hallucination_label}\n"
        markdown_output += "\n### Translated Text\n"
        markdown_output += translated_text + "\n"

        return markdown_output, audio_output
        
    except Exception as e:
        return f"Error occurred during processing: {e}. No hallucination evaluation.", None


def clear():
    # Return default values for each component
    return "English", None, None, "", None


def create_interface():
    # with gr.Blocks(theme='ParityError/Anime') as iface:
    with gr.Blocks(theme='ParityError/Anime') as interface:
        # Display the welcome message
        gr.Markdown(welcome_message)
        # Extract the full names of the languages
        language_names = list(languages.keys())

        # Add a 'None' or similar option to represent no selection
        input_language_options = ["None"] + language_names

        # Create a dropdown for language selection
        input_language = gr.Dropdown(input_language_options, label="Select the language", value="English", interactive=True)
        
        with gr.Accordion("Use Voice", open=False) as voice_accordion:
            audio_input = gr.Audio(label="Speak")
            audio_output = gr.Markdown(label="Output text")  # Markdown component for audio
            gr.Examples([["audio1.wav"], ["audio2.wav"], ], inputs=[audio_input])

        with gr.Accordion("Use a Picture", open=False) as picture_accordion:
            image_input = gr.Image(label="Upload image")
            image_output = gr.Markdown(label="Output text")  # Markdown component for image
            gr.Examples([["image1.png"], ["image2.jpeg"], ["image3.jpeg"], ], inputs=[image_input])

        with gr.Accordion("MultiMed", open=False) as multimend_accordion:
            text_input = gr.Textbox(label="Use Text", lines=3, placeholder="I have had a sore throat and phlegm for a few days and now my cough has gotten worse!")
        
            gr.Examples([
                ["What is the proper treatment for buccal herpes?"],
                ["I have had a sore throat and hoarse voice for several days and now a strong cough recently "],
                ["How does cellular metabolism work TCA cycle"],
                ["What special care must be provided to children with chicken pox?"],
                ["When and how often should I wash my hands?"],
                ["بکل ہرپس کا صحیح علاج کیا ہے؟"],
                ["구강 헤르페스의 적절한 치료법은 무엇입니까?"],
                ["Je, ni matibabu gani sahihi kwa herpes ya buccal?"],
            ], inputs=[text_input])

        text_output = gr.Markdown(label="MultiMed")  
        audio_output = gr.Audio(label="Audio Out", type="filepath")
        
        text_button = gr.Button("Use MultiMed")
        text_button.click(process_and_query, inputs=[input_language, audio_input, image_input, text_input], outputs=[text_output, audio_output])

        clear_button = gr.Button("Clear")
        clear_button.click(clear, inputs=[], outputs=[input_language, audio_input, image_input, text_output, audio_output])

    return interface


app = create_interface()
app.launch(show_error=True, debug=True)
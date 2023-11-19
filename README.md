---
title: MultiMed
emoji: ⚕️😷🦠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.0.2
app_file: app.py
pinned: true
license: mit
---
# 👋🏻Welcome to Multimed 

Multimed currently uses Fuyu , Stable LM and Seamless-m4t , for multimodal inputs and outputs. 

### **Mission**: To democratize access to reliable medical information and public health education through advanced, multilingual, and multimodal technology.

### **Vision**: To become a global leader in providing accessible, accurate, and immediate medical guidance and health education, bridging language and accessibility barriers.

### **Overview**:
MultiMed is an innovative company operating at the intersection of health technology and educational technology. It specializes in developing advanced software solutions focused on medical Q&A, public health information, and sanitation education. The company's flagship product is the MultiMed app, a highly accessible and multilingual platform designed to provide accurate medical information and public health education to a diverse global audience.

### Key Features of the MultiMed App:
- Multimodal Input: The app accepts both text and voice inputs, making it accessible to users with different preferences or disabilities. This feature includes speech-to-text capabilities, allowing users to ask questions verbally.

- **Multilingual Support**: MultiMed supports multiple languages, breaking down language barriers in accessing vital health information. This feature is crucial in serving non-English speaking populations and ensuring inclusivity.

- **Conversational AI**: The app uses advanced AI to understand and respond to user inquiries in a conversational manner. This AI is continuously updated with the latest medical research and public health guidelines.

- **Text and Speech Output**: Responses can be delivered in both text and spoken form, catering to the needs of users with visual impairments or those who prefer auditory learning.

- **Medical Q&A**: Users can ask specific medical questions and receive evidence-based answers. This feature covers a wide range of medical topics, from general health queries to more complex medical conditions.

- **Public Health and Sanitation Education**: The app provides educational content on public health and sanitation, crucial for preventing disease and promoting health, especially in underserved communities.

- **User Accessibility Features**: The app includes screen reader compatibility, high-contrast modes, and easy-to-navigate interfaces to cater to users with different abilities.

- **Regular Updates**: The app is regularly updated with the latest health guidelines and medical research, ensuring that the information provided is current and reliable.

- **Community Engagement**: Features like forums or community Q&A sessions where users can learn from others' experiences and share their own.

- **Privacy and Security**: Strong data protection measures to ensure user privacy and security, especially important given the sensitive nature of medical information.

### Target Audience:
- Individuals seeking immediate medical advice or health information.
- Non-native English speakers requiring medical information in their native language.
- People with disabilities who benefit from multimodal input and output options.
- Educational institutions and public health organizations looking for a tool to aid in health education.
- Healthcare professionals seeking a tool for patient education and engagement.
### Impact and Social Responsibility:
MultiMed is committed to social responsibility, focusing on reaching underserved communities and contributing to global health education. The company collaborates with health organizations and NGOs to ensure that accurate and vital health information is accessible to all, regardless of their location, language, or socio-economic status.

### Future Developments:
MultiMed plans to integrate more languages and dialects, expand its database to cover more specialized medical fields, and collaborate with global health experts to enhance the accuracy and relevance of its content. Additionally, the company is exploring the integration of augmented reality (AR) for more interactive health education and the potential use of blockchain for secure patient data management.

# Set Up Instructions

1 . select a folder to download then run :

```bash
git clone https://github.com/Josephrp/multi-med.git && cd multi-med
```

2. Set up Environments

```bash
python -m venv venv
```

create and edit a .env file with the following information:

![image](https://github.com/Josephrp/multi-med/assets/18212928/52bcc0f5-13dd-4b46-a196-be24ffe7f080)

for Vectara and Hugging Face Access to your own models and knowledge bases. 

3. Install Requirements

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

5. Following the instructions to find your app
or visit : ``` http://127.0.0.1:7860```

6. you can also use this app via api on your own deploymend
Example :
```python
from gradio_client import Client

client = Client("https://teamtonic-multimed.hf.space/--replicas/q9njr/")
result = client.predict(
		None,	# Literal[None, English, Modern Standard Arabic, Bengali, Catalan, Czech, Mandarin Chinese, Welsh, Danish, German, Estonian, Finnish, French, Hindi, Indonesian, Italian, Japanese, Korean, Maltese, Dutch, Western Persian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Swahili, Telugu, Tagalog, Thai, Turkish, Ukrainian, Urdu, Northern Uzbek, Vietnamese]  in 'Select the language' Dropdown component
		https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav,	# filepath  in 'Speak' Audio component
		https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png,	# filepath  in 'Upload image' Image component
		"Hello!!",	# str  in 'Use Text' Textbox component
							api_name="/process_and_query"
)
print(result)
```

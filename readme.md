# Project Rim

**Rim** is a cutting-edge AI system that integrates multiple open-source and licensed models like **LLaMA 3.2**, **GPT-2**, and **Stable Diffusion v1.4** to perform a wide variety of tasks. With dynamic intent detection and intelligent model switching, Rim excels in both text and image generation tasks, utilizing the strengths of each model effectively.

---

## Features

- **Dynamic Model Switching**: Utilizes different AI models based on the task type:
    - **LLaMA 3.2-1B-Instruct**: Handles general text generation.
    - **LLaMA 3.2-3B**: Designed for complex language tasks.
    - **GPT-2**: Provides creative and fine-tuned text outputs.
    - **Stable Diffusion v1.4**: Generates high-quality images from textual prompts.
- **Intent Detection**: Built with **spaCy**, Rim analyzes user inputs to decide whether to process tasks as text or image generation.
- **Flask-Driven API**: A lightweight backend to manage requests to multiple models.
- **CUDA Acceleration**: Leverages GPU support to ensure optimal performance during model inference.

---

## Getting Started

### Prerequisites

Ensure you have the following installed on your machine:

- **Python**: Version 3.10 or higher.
- **CUDA Toolkit**: Make sure the compatible version for your GPU is installed for acceleration.
- **GPU (Optional)**: While not required, it is highly recommended for faster performance and GPU-dependent models like Stable Diffusion.

---

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/project-rim.git
   cd project-rim
   ```

2. **Set Up a Virtual Environment**

   ```bash
   # Create a virtual environment
   python3.10 -m venv venv

   # Activate the virtual environment
   # For Linux/MacOS:
   source venv/bin/activate
   # For Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**

   Install all required Python packages using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-Trained Models**

   Download or clone the required models:
    - **LLaMA 3.2-1B-Instruct**
    - **LLaMA 3.2-3B**
    - **Stable Diffusion v1.4**
    - **GPT-2**

   Follow their respective hosting sources (e.g., Hugging Face) and organize them into your project directory accordingly.

---

### Running Rim

1. **Start the Flask Application**

   Once everything is installed, start the Flask server:

   ```bash
   python main.py
   ```

2. **Access the Application**

   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **API Interaction**

   Use the `/api/chat` endpoint to send text prompts via a POST request. Example payload:

   ```json
   {
     "prompt": "Generate an image of a futuristic city."
   }
   ```

   The response will dynamically select the appropriate model and return either text or image outputs based on the input.

---

## System Workflow

1. **Input Analysis**:
    - User inputs are processed through the `/api/chat` API or a simple web interface.

2. **Intent Detection**:
    - **spaCy NLP** detects whether the input requires text generation or image creation.

3. **Model Selection**:
    - Rim intelligently selects and routes tasks to the corresponding AI model.
        - **Text Prompts**: Routed to LLaMA or GPT-2 models.
        - **Image Prompts**: Routed to **Stable Diffusion v1.4**.

4. **Output Delivery**:
    - Text responses are returned via JSON.
    - Generated images are stored in the `static/images/` directory and served via the web app.

---

## Directory Structure

```plaintext
.
├── main.py                 # Entry point for starting the Flask app
├── inference_model.py      # Contains intent detection and model-switching logic
├── templates/              # HTML templates for the web app
│   └── index.html          # Homepage UI
├── static/                 
│   └── images/             # Folder to store generated images
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── models/                 # Folder to download/store pre-trained models
```

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## Contributing

Contributions to **Project Rim** are welcome! Here's how you can contribute:

1. **Fork this repository.**
2. **Create a new branch**:
   ```bash
   git checkout -b feature-branch
   ```
3. **Make your changes and commit**:
   ```bash
   git commit -m "Add some feature"
   ```
4. **Push to your branch**:
   ```bash
   git push origin feature-branch
   ```
5. **Submit a pull request**.

---

## Acknowledgements

This project leverages the power of the following technologies and models:

- **Transformers (Hugging Face)**: For LLaMA and GPT-2 models.
- **Diffusers**: Stable Diffusion for image generation.
- **spaCy**: Used for Natural Language Processing in intent detection.
- **CUDA**: For boosting GPU-accelerated performance.

Special thanks to the developers and contributors of **LLaMA 3.2 models**, **GPT-2**, and **Stable Diffusion** for creating these cutting-edge solutions.

---

Feel free to replace placeholders like `your-username` or `project-rim` with the appropriate details!
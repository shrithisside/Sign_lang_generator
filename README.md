#Introduction

Welcome to **Voice2Gesture**, a web application designed to make communication more inclusive. This tool converts spoken language in videos into sign language, providing visual representations of signs to displayed for deaf people. Whether you're an educator, content creator, or learner, Sign Language Helper helps bridge the gap between hearing and non-hearing communities, making information accessible to everyone with ease.

## Features

- Upload video and convert it to sign language.
- Display generated images of sign language for each letter.
- Download the processed video with captions.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2. Set up a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Apply migrations:

    ```bash
    python manage.py migrate
    ```

5. Run the development server:

    ```bash
    python manage.py runserver
    ```

## Usage

1. **Open the Application:**
Open your web browser and navigate to http://127.0.0.1:8000/.

2. **Upload a Video:**
Click on the upload button and select a video file in the video.mp4 format.

3. **Processing the Video:**
Once the video is uploaded, the application will start processing it. This involves several steps to extract and convert the audio to text, and then generate sign language images.

4. **Viewing and Downloading:**
After processing is complete, you can view the video with the sign language overlay directly in the browser. There will also be an option to download the processed video.

## How It Works

1. **Upload a Video:** Start by uploading a video file in the video.mp4 format.
2. **Audio Extraction:** Upon submission, the audio is extracted from the video using pydub.
3. **Speech-to-Text Conversion:** The extracted audio is processed to convert speech to text using NLTK.
4. **Text Processing:** The processed text is then fed into a trained model using TensorFlow and Keras.
5. **Sign Generation:** The model generates corresponding sign language images for each word in the text.
6. **Video Overlay:** These sign language images are overlaid onto the original video to display the signs along with the video.   

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m "Add new feature"`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a Pull Request.

## License

This project is licensed under the MIT License, which allows for open-source use, modification, and distribution.


## Video Demo
For a visual demonstration of the Sign Language Helper in action, you can watch the video demo. This demo walks through the entire process from video upload to the final downloadable output, showcasing the application’s features and functionality.
[Link to video demo](https://your-demo-video-link)

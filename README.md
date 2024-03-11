
Processing of Customer Service Calls

## Overview
CSPANLP is a comprehensive package designed for the Natural Language Processing (NLP) of customer service call data. This package offers a suite of tools for transcribing, redacting, summarizing, topic modeling, and sentiment analysis of customer service calls, aiming to enhance the understanding and management of customer interactions in the utility sector.

![MicrosoftTeams-image](https://github.com/shaunck96/Customer-Service-Analytics-Pipeline/assets/79271767/27d95465-d190-4a51-8028-70ade0d01f1c)


## Features
- **Audio Processing:** Transcribe customer service calls and extract meaningful information from audio data.
- **PII Redaction:** Automatically redact Personally Identifiable Information (PII) from transcriptions, ensuring data privacy and compliance.
- **Summarization:** Generate concise summaries of call contents, providing quick insights into customer interactions.
- **Topic Modeling:** Identify and categorize the main topics discussed in the calls, aiding in better customer service analysis.
- **Sentiment Analysis:** Evaluate the sentiment of customer interactions, determining whether the overall tone is positive, negative, or neutral.

## Installation
To install CSPANLP, run the following command:
```bash
pip install CSPANLP
```

## Requirements
CSPANLP requires the following packages:
- pandas
- numpy
- spacy
- transformers
- regex
- nltk
- azure-storage-blob
- huggingface_hub
- sentence_transformers
- librosa
- faster_whisper
- presidio_analyzer
- presidio_anonymizer

These dependencies will be automatically installed along with CSPANLP.

## Usage
To use CSPANLP in your project, follow these steps:

### Initialization:
Import the necessary classes from the package and initialize them with appropriate configurations.
```python
from cs_utils_with_pii_redaction import AudioProcessor, AllContextWindowSummaryGenerator, TopicGenerator, sentiment_eval
from db_utils import DBUtilConnectionCreator

# Initialize components with database and storage configurations
db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()
audio_processor = AudioProcessor(abfsClient, pytest_flag=False, db=db)
```

### Audio Transcription and Redaction:
Transcribe audio files and redact sensitive information.
```python
audio_processor.transcription_redaction_trigger()
```

### Summarization:
Generate summaries for transcribed calls.
```python
summary_generator = AllContextWindowSummaryGenerator(db=db, abfs_client=abfsClient, pytest_flag=False)
summary_generator.summary_generation_trigger()
```

### Topic Modeling:
Extract topics from the call summaries.
```python
topic_generator = TopicGenerator(db=db, abfs_client=abfsClient, pytest_flag=False)
topic_generator.topic_generator()
```

### Sentiment Analysis:
Analyze the sentiment of the calls.
```python
sentiment_analyzer = sentiment_eval(db=db, abfs_client=abfsClient, pytest_flag=False)
sentiment_analyzer.sentiment_emotion_classifier()

### LATEST UPDATES:
Llama2 chat for customer service insights
```

## License
This project is licensed under the [License Name].

## Contributing
Contributions to CSPANLP are welcome. Please refer to the CONTRIBUTING.md file for guidelines.

## Support
For support and queries, please contact [shaunshib96@gmail.com](mailto:sshibu@gmail.com).

## Authors and Acknowledgment
- Shaun Shibu - Initial work - [shaunshib96@gmail.com](mailto:sshibu@gmail.com)
- Acknowledgments to the team and contributors who supported this project.

## Additional Information
For more detailed information about each module and function in CSPANLP, please refer to the detailed documentation provided within the package.

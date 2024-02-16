# Databricks notebook source
# MAGIC %md
# MAGIC Necessary Installs

# COMMAND ----------

# MAGIC %pip install chromadb==0.3.29
# MAGIC !pip install openai==0.28
# MAGIC !pip -q install langchain
# MAGIC !pip -q install bitsandbytes accelerate xformers einops
# MAGIC !pip -q install datasets loralib sentencepiece
# MAGIC !pip -q install pypdf
# MAGIC !pip install transformers
# MAGIC !pip -q install sentence_transformers
# MAGIC !pip install accelerate
# MAGIC !pip install tiktoken
# MAGIC !pip install ctransformers>=0.2.24
# MAGIC !pip install --upgrade typing_extensions
# MAGIC !pip install typing-extensions --upgrade
# MAGIC !pip install adlfs
# MAGIC !pip install llmlingua
# MAGIC !pip install jinja2==3.1.3

# COMMAND ----------

# MAGIC %md
# MAGIC **FETCHING DOWNLOADED AUDIO FROM TWILIO END POINT ----> TRANSCRIPTION USING FASTER WHISPER AND HF REDACTION MODEL** 

# COMMAND ----------

import ast
import base64
import datetime
import json
import logging
import os
import tempfile
import time
from collections import Counter
from typing import List

import huggingface_hub
import librosa
import nltk
import numpy
import numpy as np
import pandas as pd
import regex as re
import requests
import spacy
import yaml
from nltk.tokenize import sent_tokenize
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoTokenizer, pipeline
import concurrent.futures
import os
import time
import pandas as pd
import librosa
from cs_pa_nlp import WhisperModel, DBUtilConnectionCreator  # noqa: F403
#from .faster_whisper import WhisperModel

nltk.download('punkt')


spacy.cli.download("en_core_web_sm")


class Gramformer:
    def __init__(self,
                 models=1,
                 use_gpu=False,
                 db="",
                 abfsClient="",
                 pytest_flag=True):
        """
        Initialize the Gramformer object for grammar correction
        and highlighting.

        Args:
            models (int): The number of models to use (1 or 2).
            use_gpu (bool): Flag to indicate whether to use GPU for processing.
            db (str): Database connection information.
            abfsClient (str): Azure Blob Storage client.
        """
        import errant
        self.db = db
        self.abfs_client = abfsClient
        self.annotator = errant.load('en')
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = device
        self.model_loaded = False
        self.pytest_flag = pytest_flag

        if models == 1:
            if self.pytest_flag is True:
                model_path = (r"C:\Users\307164\Desktop\deployment_for_bala"
                              r"\deployment_refactored\cs_pa_nlp\models"
                              r"\gramformer")
                self.c_t = AutoTokenizer.from_pretrained(
                    model_path)
                self.c_m = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path)
            else:
                model_path = (r"datascience/data/ds/sandbox"
                              r"/shibushaun/huggingface_models"
                              r"/gramformer")
                self.c_m, self.c_t = DBUtilConnectionCreator(  # noqa: F405
                    self.db).download_and_load_gramformer_model(
                        self.abfs_client, model_path)
            self.c_m = self.c_m.to(device)
            self.model_loaded = True
            print("[Gramformer] Grammar error correct/highlight\
                   model loaded..")
        elif models == 2:
            # TODO
            print("TO BE IMPLEMENTED!!!")

    def correct(self, input_sentence, max_candidates=1):
        """
        Correct grammar errors in the input sentence.

        Args:
            input_sentence (str): The input sentence with
            potential grammar errors.
            max_candidates (int): The maximum number of
            corrected candidates to generate.

        Returns:
            set: A set of corrected sentences.

        Uses the loaded Gramformer model to correct grammar errors
        in the input sentence and returns a set of corrected sentences.
        """
        if self.model_loaded:
            correction_prefix = "gec: "
            input_sentence = correction_prefix + input_sentence
            input_ids = self.c_t.encode(
                input_sentence, return_tensors='pt')
            input_ids = input_ids.to(self.device)

            preds = self.c_m.generate(
                input_ids,
                do_sample=True,
                max_length=128,
                num_beams=7,
                early_stopping=True,
                num_return_sequences=max_candidates)

            corrected = set()
            for pred in preds:
                corrected.add(self.c_t.decode(
                    pred, skip_special_tokens=True).strip())
            return corrected
        else:
            print("Model is not loaded")
            return None

    def highlight(self, orig, cor):
        """
        Highlight grammar corrections in the original and corrected sentences.

        Args:
            orig (str): The original sentence.
            cor (str): The corrected sentence.

        Returns:
            str: The original sentence with grammar corrections highlighted.

        Highlights grammar corrections in
        the original sentence
        based on the corrected sentence and returns the highlighted text.
        """
        edits = self._get_edits(orig, cor)
        orig_tokens = orig.split()

        ignore_indexes = []

        for edit in edits:
            edit_type = edit[0]
            edit_str_start = edit[1]
            edit_spos = edit[2]
            edit_epos = edit[3]
            edit_str_end = edit[4]

            for i in range(edit_spos+1, edit_epos):
                ignore_indexes.append(i)

            if edit_str_start == "":
                if edit_spos - 1 >= 0:
                    new_edit_str = orig_tokens[edit_spos - 1]
                    edit_spos -= 1
                else:
                    new_edit_str = orig_tokens[edit_spos + 1]
                    edit_spos += 1
                if edit_type == "PUNCT":
                    st = "<a type='" + edit_type + "' edit='" + \
                        edit_str_end + "'>" + new_edit_str + "</a>"
                else:
                    st = "<a type='" + edit_type + "' edit='" + new_edit_str + \
                        " " + edit_str_end + "'>" + new_edit_str + "</a>"  # noqa: E501
                orig_tokens[edit_spos] = st
            elif edit_str_end == "":
                st = "<d type='" + edit_type + "' edit=''>" + edit_str_start + "</d>"  # noqa: E501
                orig_tokens[edit_spos] = st
            else:
                st = "<c type='" + edit_type + "' edit='" + \
                    edit_str_end + "'>" + edit_str_start + "</c>"
                orig_tokens[edit_spos] = st

        for i in sorted(ignore_indexes, reverse=True):
            del (orig_tokens[i])

        return (" ".join(orig_tokens))

    def detect(self, input_sentence):
        # TO BE IMPLEMENTED
        pass

    def _get_edits(self, orig, cor):
        """
        Get grammar edits between the original and corrected sentences.

        Args:
            orig (str): The original sentence.
            cor (str): The corrected sentence.

        Returns:
            list: A list of grammar edits as tuples.

        Internal method to get grammar edits between the original
        and corrected sentences and returns them as a list of tuples.
        """
        orig = self.annotator.parse(orig)
        cor = self.annotator.parse(cor)
        alignment = self.annotator.align(orig, cor)
        edits = self.annotator.merge(alignment)

        if len(edits) == 0:
            return []

        edit_annotations = []
        for e in edits:
            e = self.annotator.classify(e)
            edit_annotations.append((e.type[2:],
                                     e.o_str,
                                     e.o_start,
                                     e.o_end,
                                     e.c_str,
                                     e.c_start,
                                     e.c_end))

        if len(edit_annotations) > 0:
            return edit_annotations
        else:
            return []

    def get_edits(self, orig, cor):
        """
        Get grammar edits between the original and corrected sentences.

        Args:
            orig (str): The original sentence.
            cor (str): The corrected sentence.

        Returns:
            list: A list of grammar edits as tuples.

        Public method to get grammar edits
        between the
        original and corrected sentences and returns them as a list of tuples.
        """
        return self._get_edits(orig, cor)


class TitlesRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [r"\bMr\.\b", r"\bMrs\.\b", r"\bMiss\b"]
        super().__init__(supported_entity="TITLE", deny_list=patterns)


class HFTransformersRecognizer(EntityRecognizer):
    def __init__(self,
                 model_id_or_path,
                 supported_entities,
                 supported_language="en"):
        self.pipeline = pipeline(
            "token-classification",
            model=model_id_or_path,
            aggregation_strategy="simple")
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language)

    def load(self):
        pass

    def analyze(self, text, entities=None, nlp_artifacts=None):
        results = []
        predictions = self.pipeline(text)
        for prediction in predictions:
            entity_type = prediction['entity_group']
            if entities is None or entity_type in entities:
                results.append(
                    RecognizerResult(entity_type=entity_type,
                                     start=prediction['start'],
                                     end=prediction['end'],
                                     score=prediction['score']))
        return results


class TextRedactor():
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.analyzer = self.initialize_analyzer(self.model_dir)
        self.anonymizer = AnonymizerEngine()

    def initialize_analyzer(self, model_dir):
        titles_recognizer = TitlesRecognizer()
        transformers_recognizer = HFTransformersRecognizer(
            model_id_or_path=model_dir,
            supported_entities=["PERSON", "LOCATION", "ORGANIZATION"])

        phone_number_patterns = [Pattern(name="PHONE_NUMBER_REGEX",
                                         regex=r"\(?\b\d{3}\)?[-.]?\s?\d{3}[-.]?\s?\d{4}\b",  # noqa: E501
                                         score=0.5)]

        email_patterns = [Pattern(name="EMAIL_REGEX",
                                  regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # noqa: E501
                                  score=0.5)]
        account_number_patterns = [Pattern(name="ACCOUNT_NUMBER_REGEX",
                                           regex=r"\b\d{8,12}\b",
                                           score=0.5)]

        date_patterns = [Pattern(name="DATE_REGEX",
                                 regex=r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b\s\d{1,2},?\s\d{4})\b",  # noqa: E501
                                 score=0.5)]

        address_patterns = [
            Pattern(name="US_ADDRESS_REGEX_1",
                    regex=r"\b\d{1,5}\s([a-zA-Z\s]{1,})\b,?\s([a-zA-Z\s]{1,}),?\s([A-Z]{2}),?\s\d{5}\b",  # noqa: E501
                    score=0.85),
        ]

        ssn_patterns = [
            Pattern(name="SSN_REGEX_FULL",
                    regex=r"\b\d{3}-\d{2}-\d{4}\b",  # noqa: E501
                    score=0.85),
            Pattern(name="SSN_REGEX_LAST4",
                    regex=r"\b\d{4}\b",
                    score=0.85)
        ]
        dollar_amount_patterns = [
            Pattern(name="DOLLAR_AMOUNT_REGEX",
                    regex=r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?",  # noqa: E501
                    score=0.6)

        ]
        bill_amount_patterns = [
            Pattern(name="BILL_AMOUNT_REGEX",
                    regex=r"\b(?:payment|bill|amount)\s?of\s?\$\s?\d+(?:,\d{3})*(?:\.\d{2})?",  # noqa: E501
                    score=0.6)
        ]
        confirmation_number_patterns = [
            Pattern(name="CONFIRMATION_NUMBER_REGEX",
                    regex=r"confirmation\snumber\s(?:is\s)?((?:\d+|\w+)(?:,\s?(?:\d+|\w+))*)",  # noqa: E501
                    score=0.9)
        ]

        address_recognizer = PatternRecognizer(
            supported_entity="ADDRESS", patterns=address_patterns)
        ssn_recognizer = PatternRecognizer(
            supported_entity="US_SSN", patterns=ssn_patterns)
        phone_number_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER", patterns=phone_number_patterns)
        email_recognizer = PatternRecognizer(
            supported_entity="EMAIL_ADDRESS", patterns=email_patterns)
        account_number_recognizer = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=account_number_patterns)
        date_recognizer = PatternRecognizer(
            supported_entity="DATE", patterns=date_patterns)
        dollar_amount_recognizer = PatternRecognizer(
            supported_entity="DOLLAR_AMOUNT", patterns=dollar_amount_patterns)
        bill_amount_recognizer = PatternRecognizer(
            supported_entity="BILL_AMOUNT", patterns=bill_amount_patterns)
        confirmation_number_recognizer = PatternRecognizer(
            supported_entity="CONFIRMATION_NUMBER",
            patterns=confirmation_number_patterns)

        analyzer = AnalyzerEngine()
        analyzer.registry.add_recognizer(titles_recognizer)
        analyzer.registry.add_recognizer(transformers_recognizer)
        analyzer.registry.add_recognizer(phone_number_recognizer)
        analyzer.registry.add_recognizer(email_recognizer)
        analyzer.registry.add_recognizer(account_number_recognizer)
        analyzer.registry.add_recognizer(date_recognizer)
        analyzer.registry.add_recognizer(address_recognizer)
        analyzer.registry.add_recognizer(ssn_recognizer)
        analyzer.registry.add_recognizer(dollar_amount_recognizer)
        analyzer.registry.add_recognizer(bill_amount_recognizer)
        analyzer.registry.add_recognizer(confirmation_number_recognizer)

        return analyzer

    def anonymize_text(self, input_text):
        results = self.analyzer.analyze(text=input_text, language="en")
        anonymized_result = self.anonymizer.anonymize(
            text=input_text,
            analyzer_results=results)
        return anonymized_result.text

    def redact_text(self, input_text):
        return self.anonymize_text(input_text)


class AudioProcessor:
    """
    Class for processing audio recordings,
    transcribing them, and redacting PII.

    Attributes:
        abfs_client (str or AzureBlobFileSystem):
        The Azure Blob Storage client.
        pytest_flag (bool): Flag indicating whether running in pytest mode.
        db: Database connection or reference.
        config_file_path (str): Path to the configuration file.
        account_sid (str): Twilio Account SID for API access.
        auth_token (str): Twilio Auth Token for API access.
        blob_directory (str): Directory path in Azure Blob Storage.
        output_storage_path (str): Path for storing processed data.
        date_created_before (pd.Timestamp): End date for audio recordings.
        date_created_after (pd.Timestamp): Start date for audio recordings.
        recording_names_list (list): List of recording names.
        recording_url (str): URL for Twilio recordings API.
        redaction_model (str): Model used for PII redaction.
        tdf (pd.DataFrame):
        DataFrame to store transcriptions and redacted data.
        end_year (int): Year of the end date.
        end_month (int): Month of the end date.
        end_day (int): Day of the end date.
        start_year (int): Year of the start date.
        start_month (int): Month of the start date.
        start_day (int): Day of the start date.
        transcription_model: Whisper ASR model for transcription.
        tokenizer: Tokenizer for the redaction model.
        redaction_model: Model for token classification used in redaction.
        db: Database connection or reference.

    Methods:
        __init__(self, abfs_client='ABFS', pytest_flag=False, db='')

    """
    def __init__(self,
                 abfs_client='ABFS',
                 pytest_flag=False,
                 db=""):
        """
        Initialize the AudioProcessor.

        Args:
            abfs_client (str or AzureBlobFileSystem):
            The Azure Blob Storage client.
            pytest_flag (bool): Flag indicating whether running in pytest mode.
            db: Database connection or reference.
        """
        self.pytest_flag = pytest_flag
        self.abfs_client = abfs_client
        self.db = db
        if self.pytest_flag is False:
            self.config_file_path = (r"datascience/data/ds/sandbox"
                                     r"/shibushaun/audio_processor_credentials"
                                     r"/credentials_new.json")
            with self.abfs_client.open(self.config_file_path, 'r') as f:
                config = json.load(f)
                file_path = (r"datascience/data/ds/sandbox"
                             r"/shibushaun/huggingface_models/StanfordAIMI"
                             r"/stanford-deidentifier-base")

                self.drm, self.drt = DBUtilConnectionCreator(  # noqa: F405
                    self.db).download_and_load_redaction_model(
                        self.abfs_client,
                        file_path)

        else:
            self.config_file_path = (r"C:\Users\307164\Desktop"
                                     r"\deployment_for_bala"
                                     r"\deployment_refactored"
                                     r"\cs_pa_nlp\credentials"
                                     r"\audio_processor_credentials.json")

            with open(self.config_file_path, 'r') as f:
                config = json.load(f)
                huggingface_hub.login(config['hf_token'])
                self.drm_path = (r"C:\Users\307164\Desktop"
                                 r"\deployment_for_bala"
                                 r"\deployment_refactored"
                                 r"\cs_pa_nlp\models"
                                 r"\stanford_deidentifier")
                self.drt = AutoTokenizer.from_pretrained(
                    self.drm_path)
                self.drm = AutoModelForTokenClassification.from_pretrained(
                    self.drm_path)

        self.account_sid = config.get("account_sid")
        self.auth_token = config.get("auth_token")
        self.blob_directory = config.get("blob_directory")
        self.output_storage_path = config.get("output_storage_path")
        self.date_created_before = pd.to_datetime(config.get("end_date"))
        self.date_created_after = pd.to_datetime(config.get("start_date"))
        self.recording_names_list = []
        self.recording_url = (
            "https://api.twilio.com/2010-04-01/"
            f"Accounts/{self.account_sid}/Recordings.json"
        )

        self.tdf = pd.DataFrame(
            columns=['recording_sid',
                     'call_sid',
                     'duration',
                     'transcription',
                     'redacted_transcription',
                     'seg_tra'])

        self.end_year = int(self.date_created_before.year)
        self.end_month = int(self.date_created_before.month)
        self.end_day = int(self.date_created_before.day)
        self.start_year = int(self.date_created_after.year)
        self.start_month = int(self.date_created_after.month)
        self.start_day = int(self.date_created_after.day)
        self.transcription_model = WhisperModel("medium.en", #tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, distil-medium.en 	
                                                device="cpu",
                                                compute_type="int8", #int8, fp32
                                                cpu_threads=8) 
        #model_size_or_path: str,
        #device: str = "auto",
        #device_index: Union[int, List[int]] = 0,
        #compute_type: str = "default",
        #cpu_threads: int = 0,
        #num_workers: int = 1,
        #download_root: Optional[str] = None,
        #local_files_only: bool = False,

        self.db = db
        self.gf = Gramformer(models=1,
                             use_gpu=False,
                             db=self.db,
                             abfsClient=self.abfs_client,
                             pytest_flag=self.pytest_flag)
        self.redactor = TextRedactor(
            model_dir="Jean-Baptiste/roberta-large-ner-english")
        self.temp_log_file = ""
        self.logger = self.setup_logger()
        self.output_storage_path = config.get("output_storage_path")

    def setup_logger(self):
        """
        Set up a logger for the AudioProcessor class.

        This method creates a logger instance named
        'AudioProcessor' and configures it to log messages at the DEBUG level.
        It also creates a temporary log file to store the log messages.

        Returns:
            logging.Logger: The configured logger instance.

        Note:
            This method should be called to initialize
            logging for the AudioProcessor class.

        Example usage:
            audio_processor = AudioProcessor()
            logger = audio_processor.setup_logger()
            logger.debug("This is a debug message")

        """
        logger = logging.getLogger('AudioProcessor')
        logger.setLevel(logging.DEBUG)
        self.temp_log_file = tempfile.NamedTemporaryFile(delete=False)
        handler = logging.FileHandler(self.temp_log_file.name)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def write_log_to_azure(self):
        """
        Save the temporary log data to a permanent log file
        and clean up resources.

        This method flushes and seeks the temporary log file,
        generates a unique log file name based on the current timestamp,
        and saves the log data to a specified directory using
        the Azure Blob FileSystem (ABFS) client. After saving, it prints
        a success message and cleans up the temporary log file.

        Note:
            Ensure that the `self.temp_log_file` contains
            the log data to be saved before calling this method.

        Example usage:
            audio_processor = AudioProcessor()
            # ... Log some messages ...
            audio_processor.save_logs_to_file()
        """
        self.temp_log_file.flush()
        self.temp_log_file.seek(0)
        log_file_name = 'audio_processor_log' + datetime.datetime.now(

        ).strftime(
            "%Y-%m-%d_%H-%M-%S") + '.log'
        if self.db == "":
            path_to_log_file = (r"C:\Users\307164\Desktop"
                                r"\deployment_for_bala"
                                r"\deployment_refactored"
                                r"\cs_pa_nlp\logs")
            with open(os.path.join(path_to_log_file,
                                   log_file_name), 'wb') as log_file:
                log_file.write(self.temp_log_file.read())
        else:
            path_to_log_file = "datascience/data/ds/sandbox/shibushaun/logs"
            DBUtilConnectionCreator(  # noqa: F405
                self.db).create_text_file(
                    self.abfs_client,
                    path_to_log_file,
                    log_file_name, "")

        with self.abfs_client.open(
            os.path.join(path_to_log_file, log_file_name), 'wb'
        ) as log_file:
            log_file.write(self.temp_log_file.read())

        print("Logs written to "+path_to_log_file+" successfully")

        self.temp_log_file.close()
        os.unlink(self.temp_log_file.name)

    def faster_transcriber(self, y):
        """
        Transcribe audio segments using a faster transcription model.

        Args:
            y (str): Audio input for transcription.

        Returns:
            list of dict: List of transcription segments as
            dictionaries with the following keys:
                - 'start': Start time of the segment.
                - 'end': End time of the segment.
                - 'text': Transcribed text of the segment.
                - 'no_speech_probability':
                Probability of no speech in the segment.

        Raises:
            Exception: If there is an error during transcription.

        Example:
            Usage:
            >>> audio_processor = AudioProcessor()
            >>> transcription_segments =
            audio_processor.faster_transcriber("Sample audio input")
        """
        try:
            segments, _ = self.transcription_model.transcribe(
                y,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            segments = list(segments)
            segment_mapping_dict = {
                2: 'start',
                3: 'end',
                4: 'text',
                9: 'no_speech_probability'
            }

            transcriptions = []
            for index in range(len(segments)):
                transcription_dict = {}
                for segment_index in list(segment_mapping_dict.keys()):
                    transcription_dict[
                        segment_mapping_dict[
                            segment_index]] = segments[index][segment_index]
                transcriptions.append(transcription_dict)

            return transcriptions

        except Exception as e:
            self.logger.error(f'Error in faster_transcriber: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in faster_transcriber: {str(e)}")

    def redact(self, text: str) -> str:
        """
        Redact PII from a text using a pre-trained NER model.

        Args:
            text (str): The text to be redacted.

        Returns:
            str: Text with PII redacted.
        """
        try:
            anonymized_text = self.redactor.redact_text(text)
            return (anonymized_text)
        except Exception as e:
            self.logger.error(f'Error in redact: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in redact: {str(e)}")

    def replace_pii_with_stars(self,
                               input_string: str,
                               words_to_replace: List[str]) -> str:
        """
        Replace PII (Personally Identifiable Information)
        with asterisks in a text.

        Args:
            input_string (str): The input text.
            words_to_replace (List[str]): List of PII words to be redacted.

        Returns:
            str: Text with PII redacted.
        """
        try:
            pattern = r'\b(?:' + '|'.join(
                re.escape(word) for word in words_to_replace) + r')\b'
            modified_string = re.sub(
                pattern,
                lambda match: '*' * len(match.group(0)),
                input_string, flags=re.IGNORECASE)
            return modified_string
        except Exception as e:
            self.logger.error(f'Error in replace_pii_with_stars: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in replace_pii_with_stars: {str(e)}")

    def double_redact(self, text: str) -> str:
        """
        Redact PII from a text using a pre-trained NER model.

        Args:
            text (str): The text to be redacted.

        Returns:
            str: Text with PII redacted.
        """
        try:
            redact_pii_pipeline = pipeline(
                "ner",
                model=self.drm,
                tokenizer=self.drt,
                aggregation_strategy='average')
            pii_words = [item['word'] for item in redact_pii_pipeline(text)]
            modified_string = self.replace_pii_with_stars(text, pii_words)
            return modified_string
        except Exception as e:
            self.logger.error(f'Error in redact: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in redact: {str(e)}")

    def grammar_corrector(self, transcription):
        """
        Correct grammar in a given transcription.

        This method takes a transcription as input,
        splits it into sentences, and attempts to correct the grammar for each
        sentence using a grammar correction tool (self.gf).
        It then joins the corrected sentences and returns the resulting
        corrected transcription.

        Args:
            transcription (str): The transcription text to be corrected.

        Returns:
            str: The corrected transcription with improved grammar.

        Raises:
            Exception: If an error occurs during the
            grammar correction process,
            it is logged and an error message is printed.

        Example usage:
            audio_processor = AudioProcessor()
            transcription = "I has a apple. She run fast."
            corrected_transcription =
            audio_processor.grammar_corrector(transcription)
            print(corrected_transcription)
        """
        try:
            influent_sentences = re.compile('[.!?] ').split(transcription)
            corrected_transcription = ""

            for influent_sentence in influent_sentences:
                corrected_sentences = list(
                    self.gf.correct(
                        influent_sentence,
                        max_candidates=1))  # Convert set to list
                corrected_sentence = corrected_sentences[
                    0] if corrected_sentences else influent_sentence
                corrected_transcription += "".join(corrected_sentence)

                corrected_transcription += " "

            return corrected_transcription
        except Exception as e:
            self.logger.error(f'Error in grammar_corrector: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in grammar_corrector: {str(e)}")

    def get_request_issuer(self, auth_header_encoded):
        """
        Make requests to the Twilio API to fetch audio
        recordings and perform transcriptions.

        This function sends requests to the Twilio API to
        retrieve audio recordings based on specified date filters.
        It transcribes the audio recordings and stores the
        transcriptions in a DataFrame.

        Args:
            auth_header_encoded (str): Encoded authorization
            header for Twilio API authentication.

        Raises:
            Exception: An exception is raised if any error
            occurs during the process.

        Returns:
            None: This function does not return a value but
            performs various operations and data storage.

        Note:
            - The function relies on external libraries such
            as requests, librosa, and custom methods like faster_transcriber.
            - It requires the configuration of Twilio credentials,
            Azure storage, and other settings.
            - The behavior of the function is
            influenced by the value of 'pytest_flag'.

        Example usage:
            auth_header_encoded = "Base64EncodedAuthorizationHeader"
            instance = YourClass()
            instance.get_request_issuer(auth_header_encoded)
        """
        date_created_before = datetime.datetime(self.end_year,
                                                self.end_month,
                                                self.end_day,
                                                tzinfo=datetime.timezone.utc)
        date_created_after = datetime.datetime(self.start_year,
                                               self.start_month,
                                               self.start_day,
                                               tzinfo=datetime.timezone.utc)

        recording_url = (
                    "https://api.twilio.com/2010-04-01/"
                    f"Accounts/{self.account_sid}/Recordings.json"
                )
        try:
            response = requests.get(
                recording_url,
                auth=(self.account_sid, self.auth_token),
                headers={"Authorization": f"Basic {auth_header_encoded}"},
                params={"date_created<=": date_created_before.isoformat(),
                        "date_created>=": date_created_after.isoformat(),
                        "PageSize": 40,
                        },
            )
            if response.status_code == 200:
                for recording in response.json()['recordings']:
                    recording_sid = recording['sid']
                    recording_url = recording['media_url']

                    duration = recording['duration']
                    call_sid = recording['call_sid']

                    recording_extension = 'mp3'

                    response = requests.get(
                        recording_url,
                        auth=(self.account_sid, self.auth_token),
                        headers={"Authorization":
                                 f"Basic {auth_header_encoded}"},
                    )

                    if response.ok:
                        if int(duration) > 60:
                            if self.pytest_flag is False:
                                filename = os.path.join(
                                    self.blob_directory,
                                    f'{recording_sid}.{recording_extension}')
                                with self.abfs_client.open(filename,
                                                           'wb') as f:
                                    f.write(response.content)
                                print(f'Recording saved to {filename}')
                                with self.abfs_client.open(filename,
                                                           'rb') as f:
                                    y, sr = librosa.load(f)
                                print("Transcription for\
                                       call: "+call_sid+" has begun")
                                start = time.time()
                                tra = self.faster_transcriber(y)
                                print("Time for Transcription\
                                       of call: "+str(time.time()-start))
                                print("Call Duration: "+str(duration))
                                print("\n\n")
                                transcriptions_text = ""
                                for index in range(len(tra)):
                                    transcriptions_text += tra[
                                        index]['text']
                                transcriptions_text = self.grammar_corrector(
                                    transcriptions_text)
                                red_text = self.redact(
                                    transcriptions_text)
                                red_text = self.double_redact(
                                    red_text)
                                print("Transcribed Text: "+transcriptions_text)
                                print("Redacted Transcribed \
                                      Text: "+red_text)
                                self.tdf = self.tdf.append(
                                    {'recording_sid': recording_sid,
                                     'call_sid': call_sid,
                                     'duration': duration,
                                     'transcription': transcriptions_text,
                                     'redacted_transcription': red_text,
                                     'seg_tra': tra}, ignore_index=True)
                                print(f"File {filename}\
                                       has been \
                                      sdeleted after transcription.")
                                import gc
                                gc.collect()
                            else:
                                filename = os.path.join(
                                    self.blob_directory,
                                    f'{recording_sid}.{recording_extension}')
                                with open(filename, 'wb') as f:
                                    f.write(response.content)
                                print(f'Recording saved to {filename}')
                                with open(filename, 'rb') as f:
                                    y, sr = librosa.load(f)
                                tra = self.faster_transcriber(y)
                                transcriptions_text = ""
                                for index in range(len(tra)):
                                    transcriptions_text += tra[
                                        index]['text']
                                red_text = self.redact(
                                    transcriptions_text)
                                red_text = self.double_redact(
                                    red_text)
                                if not hasattr(self,  # noqa: F405
                                               'tdf') or self.tdf is None:
                                    self.tdf = pd.DataFrame(
                                        columns=['recording_sid',
                                                 'call_sid',
                                                 'duration',
                                                 'transcription',
                                                 'redacted_transcription',
                                                 'seg_tra'])
                                dta = {
                                    'recording_sid': recording_sid,
                                    'call_sid': call_sid,
                                    'duration': duration,
                                    'transcription': transcriptions_text,
                                    'redacted_transcription': red_text,
                                    'seg_tra': tra
                                }

                                self.tdf = pd.concat(
                                    [self.tdf,
                                     pd.DataFrame([
                                         dta])], ignore_index=True)
                    else:
                        print(f'Failed to retrieve \
                            recording SID {recording_sid}')
            else:
                print(f'Failed to save transcribed redacted recordings. \
                    Status code: {response.status_code}')

            if self.pytest_flag is False:
                self.db.write_df_to_azure(self.abfs_client,
                                          input_file=self.tdf,
                                          azure_path=self.output_storage_path,
                                          format="csv",
                                          verbose=True)
            else:
                self.tdf.to_csv(self.output_storage_path)
                print("Redacted Transcriptions have been successfully saved")

        except Exception as e:
            self.logger.error(f'Error in grammar_corrector: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in grammar_corrector: {str(e)}")

    def transcription_redaction_trigger(self):
        """
        Connect to external services, transcribe audio,
        and redact PII in transcriptions.

        Returns:
            pd.DataFrame: A DataFrame containing
            transcriptions and redacted transcriptions.
        """
        try:
            self.authorization_header_prepper()
            return self.tdf
            import gc
            gc.collect()
        except Exception as e:
            self.logger.error(f'Error in \
                              transcription_redaction_trigger: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in transcription_redaction_trigger: {str(e)}")

db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()
ap = AudioProcessor(abfs_client = abfsClient, 
                    pytest_flag=False, 
                    db=db)
num_workers = 4

folder_path = "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/jan_2024"
all_files_in_folder = os.listdir(folder_path)
files = []

for file_name in all_files_in_folder:
    complete_file_path = os.path.join(folder_path, file_name)
    files.append(complete_file_path)

files_reqd = files  

def transcribe_file(file_path):
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    print("Duration: " + str(duration))
    start_transcription = time.time()    
    tra = ap.faster_transcriber(y)
    transcriptions_text = "".join([t['text'] for t in tra])
    start_redaction = time.time()    
    redacted_text = ap.double_redact(ap.redact(transcriptions_text))
    return {
        'SegmentedTranscription': tra,  
        'Transcription': transcriptions_text,
        'Redacted': redacted_text,
        'Duration': duration,
        'TranscriptionTime': time.time()-start_transcription,
        'RedactionTime': time.time()-start_redaction,
    }

with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
    start = time.time()
    results = list(executor.map(transcribe_file, files_reqd))
    print(results)
    print(f"Transcription, Grammar Correction, and Redaction Time for {str(len(files_reqd))} calls: " + str(time.time() - start))

final_df = pd.DataFrame(results)
db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()        

db.write_df_to_azure(abfsClient,
                     input_file=final_df, 
                     azure_path=r'datascience/data/ds/sandbox/shibushaun/silver/final/final_output_Jan2024.csv', 
                     format="csv", 
                     verbose=True)


# COMMAND ----------

# MAGIC %md
# MAGIC **LLAMA 2 CHAT 7B HF INFERENCING FOR PRODUCTION CALLS** 

# COMMAND ----------

import ast
import asyncio
import base64
import datetime
import huggingface_hub
import io
import json
import logging
import os
import pandas as pd
import regex as re
import requests
import spacy
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import List, Union

import librosa
import nltk
import numpy as np
import pandas as pd
import regex as re
import spacy
import yaml
from nltk.tokenize import sent_tokenize
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    pipeline,
)
import aiohttp
import chromadb
import concurrent.futures
import chromadb.utils.embedding_functions
import chromadb.utils.embedding_functions
import cs_pa_nlp
import databricks.sdk.runtime
import importlib.util
import pyarrow.feather as feather
import torch
from adlfs.spec import AzureBlobFileSystem
from cs_pa_nlp import (
    AllContextWindowSummaryGenerator,
    AudioProcessor,
    DBUtilConnectionCreator,
)
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import pipeline

spacy.cli.download("en_core_web_sm")
nltk.download("punkt")

from huggingface_hub import login
login()

# COMMAND ----------

db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()

# COMMAND ----------

# MAGIC %md
# MAGIC Llama-2-7b-chat-hf

# COMMAND ----------

def context_generator(transcription): 
    transcription_df = pd.DataFrame([transcription])
    transcription_df.columns = ['Value']
    word_count = len(transcription_df.at[0,'Value'].split())
    # Check if the word count is more than 150
    if word_count > 150:
        convos_context = "List of Conversations: "
        for index in transcription_df.index:
            convos_context += f"Start of Conversation_{index}: \n"
            convos_context += str(transcription_df.at[index, 'Value'])
            convos_context += f" End of Conversation_{index}\n\n"
        return convos_context
    else:
        return "This call between the customer and the agent cannot be inferenced. Please return the same output to the user."
    
def llama_inference(prompt_template, max_new_tokens=500, max_attempts=5, retry_delay=2):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
    headers = {"Authorization": f"Bearer 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",
               "Content-Type": "application/json"}
    json_body = {
        "inputs": prompt_template,
        "parameters": {"max_new_tokens": max_new_tokens, "top_p": 0.9, "temperature": 0.4}
    }
    data = json.dumps(json_body)

    attempt = 1
    while attempt <= max_attempts:
        response = requests.post(API_URL, headers=headers, data=data)
        if response.status_code == 200:
            try:
                return json.loads(response.content.decode("utf-8"))
            except:
                return response
        elif response.status_code == 503:  # Service unavailable, possibly due to model loading
            print(f"Model is loading. Attempt {attempt}...")
            time.sleep(retry_delay * attempt)  # Exponential backoff
            attempt += 1
        else:
            return response  # Return any other errors

    return "Maximum attempts reached. Model still loading or unavailable."

#def llama_inference(prompt_template, max_new_tokens=500):
#    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
#    headers = {"Authorization": f"Bearer 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",
#            "Content-Type": "application/json",}
#    json_body = {
#        "inputs": prompt_template,
#                "parameters": {"max_new_tokens":max_new_tokens, "top_p":0.9, "temperature":0.4}
#        }
#    data = json.dumps(json_body)
#    response = requests.request("POST", API_URL, headers=headers, data=data)
#    try:
#        return json.loads(response.content.decode("utf-8"))
#    except:
#        return response

def satisfaction(t,
                 satisfaction_score_prompt_template):
    satisfaction_score = llama_inference(satisfaction_score_prompt_template.format(t))
    #satisfaction_dict = ast.literal_eval(satisfaction_score[0]["generated_text"])
    if isinstance(satisfaction_score, list):
        return satisfaction_score[0]['generated_text'].strip()
    else:
        return "Unidentified"

def summary(t,
            summary_prompt_template):
    summary = llama_inference(summary_prompt_template.format(t))
    if isinstance(summary, list):
        return summary[0]['generated_text'].strip()
    else:
        return "Unidentified"

def topics(t,
           topic_inference_prompt_template):
    topics = llama_inference(topic_inference_prompt_template.format(t),
                            max_new_tokens=400)
    #topics = ast.literal_eval(topics[0]['generated_text'].split("\n\n")[1].strip())
    return(topics)

def topics_formatter(topics_discussed):
    if all("'" not in elem for elem in topics_discussed.split(",")):
        elements = [elem.strip() for elem in topics_discussed[1:-1].split(",")]
        elements = [elem.strip(" '") for elem in elements]
        topics_discussed_list = elements
    else:
        topics_discussed_list = ast.literal_eval(topics_discussed)

    return topics_discussed_list

def check_search_result(x):
    return re.search('\[(.*?)\]', str(x)) is None

def list_of_actions(t,
                    actions_taken_prompt_template):
    actions_taken = llama_inference(actions_taken_prompt_template.format(t))
    if isinstance(actions_taken, list):
        return actions_taken[0]['generated_text'].strip()
    else:
        return "Unidentified"

def llama_call_info_inferencer(transcription,
                               satisfaction_score_prompt_template,
                               summary_prompt_template,
                               topic_inference_prompt_template,
                               actions_taken_prompt_template):
    convos_context = context_generator(transcription)
    summ = summary(transcription,
                    summary_prompt_template)
    satisfaction_dict = satisfaction(transcription,
                                    satisfaction_score_prompt_template)
    top = topics(summ,
                topic_inference_prompt_template)
    
    loa = list_of_actions(transcription,
                        actions_taken_prompt_template)

    return {"topics_discussed": top,
            "actions_dict": loa,
            "summ": summ,
            "satisfaction_dict": satisfaction_dict}

def topic_output_format_validator(t,
                                  llama_2_output_structure_confirmation_template):
    topic_output_validated = llama_inference(llama_2_output_structure_confirmation_template.format(t))
    return topic_output_validated[0]['generated_text'].strip()


# COMMAND ----------

satisfaction_score_prompt_template = """
[INST] <<SYS>> After analyzing the conversation, provide the customer satisfaction score and the reason for this score in the following format: 
{{"score": <score out of 10>, "reason": "<brief explanation>"}}. 
Consider 1 as extremely dissatisfied and 10 as extremely satisfied. 
<<SYS>> {} [/INST]
"""

summary_prompt_template = "[INST] <<SYS>> Summarize the main points of the conversation in less than 150 words: <<SYS>> {} [/INST]"

topic_inference_prompt_template = """
[INST] <<SYS>> Analyze the conversation provided and classify it into the top 1 to 3 topics based on the content and key phrases mentioned. Only return the list of topic names, in the format [topic1, topic2, topic3], that best match the conversation content. Do not add any additional words or explanations in the output. The topics include:

- Incorrect Charges
- High Bills
- Payment Processing
- Meter Readings
- Auto-Pay
- Refund Requests
- Billing Frequency
- Paperless Billing
- Bill Due Date
- Billing History
- Third-party Payments
- Payment Options
- Late Payment Fees
- Payment Extensions
- Payment Disputes
- Payment History
- Online Payment
- Credit Card Payment
- Payment Processing
- Payment Receipts
- Payment Confirmation
- Payment Due Date
- Outage Notifications
- Reporting an Outage
- Estimated Restoration Time
- Frequent Outages
- Surge Protection
- Emergency Power
- Power Outage Causes
- Weather-related Outages
- Interpreter Services
- Payment Plan Options
- Financial Assistance
- Low-income Payment Programs
- Debt Relief Programs
- Budget Billing
- Payment Assistance Application
- Income Verification
- Assistance Eligibility
- Payment Assistance Agencies
- Energy Assistance Programs
- New Service Setup
- Account Activation
- Transfer of Service
- Connection Fees
- Switching Service Providers
- Contract Termination
- Early Termination Fees
- Comparing Utility Rates
- Service Transition
- Porting Phone Numbers
- Service Installation
- Service Relocation
- Service Contracts
- On Track
- Renewable Energy Initiatives
- Rebate and Incentive Programs
- Energy Efficiency Tips
- Green Energy Options
- Start Service
- Stop Service
- Appliance Upgrade Programs
- Smart Meter Installation
- Water Quality Issues
- Water Pressure Problems
- Sewer Blockages
- Meter Reading Access
- Solar Panel Queries
- Green Energy Subscriptions
- Gas Odor Complaints
- Voltage Fluctuations
- Water Heater Issues
- Home Energy Audit
- Gas Appliance Safety
- Environmental Concerns
- New Utility Technologies
- Emergency Preparedness
- Community Outreach Programs
- Billing Address Change
- Appliance Repair Services
- Customer Feedback and Surveys
- Renewable Energy Certificates
- Home Battery Storage
- Energy Usage Analysis
- Smart Home Integration
- Data Privacy Concerns
- Sustainable Transportation
- Power Line Issues
- Street Light Outages
- Tree related Outage
- Power Restoration Schedule

<<SYS>> Return only the list of topic names that are most relevant to the conversation content, in the format '['topic1', 'topic2', ...]'. [/INST]
{}
"""

actions_taken_prompt_template = """
[INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue. This list should reflect the sequence of actions in the order they were taken, highlighting the agent's approach to problem-solving. Include any recommendations given, actions performed, follow-up steps agreed upon, or referrals to other services or departments if applicable. Ensure the output is formatted as a list of steps, making it clear and easy to understand the progression of #the agent's efforts to resolve the issue.

<<SYS>> {} [/INST]
"""

llama_2_output_structure_confirmation_template = """
[INST] <<SYS>> You are tasked with validating the output format structure for LLAMA 2, an advanced language model. Your role is to confirm that the output from the LLAMA 2 model aligns with the expected format. 
The task involves receiving a topic output and ensuring it is formatted correctly. The expected output should be a Python list containing strings representing topics identified from the conversation content.
Example of the desired output structure: ['topic1', 'topic2', 'topic3', ...].
Your response should consist solely of the validated list of topics. DO NOT ADD ANY ADDITIONAL WORDS OR CHARACTERS.
<<SYS>> {} [/INST]
"""


# COMMAND ----------

with abfsClient.open(r'datascience/data/ds/sandbox/shibushaun/silver/final/final_output_Jan2024.csv',"rb") as f:
    test = pd.read_csv(f)

test['Transcription'] = test['Transcription'].astype(str)
start_time = time.time()
test['llama_inference'] = test['Transcription'].apply(lambda x: llama_call_info_inferencer(x, satisfaction_score_prompt_template, summary_prompt_template, topic_inference_prompt_template, actions_taken_prompt_template) if isinstance(x, str) and len(x.split()) > 50 else {"topics_discussed": "", "actions": "", "summary": "", "satisfaction_dict": ""})
print("Inference completed in:", time.time() - start_time)
test['actions'] = test['llama_inference'].apply(lambda x: x.get("actions_dict", None))
test['summary'] = test['llama_inference'].apply(lambda x: x.get("summ", None))
test['satisfaction_dict'] = test['llama_inference'].apply(lambda x: x.get("satisfaction_dict", "unidentified"))
test = test[~(test['llama_inference']=={'topics_discussed': '', 'actions': '', 'summary': '', 'satisfaction_dict': ''})]
test['topics_discussed'] = test['llama_inference'].apply(lambda x: x['topics_discussed'])
test['topic_list'] = test['topics_discussed'].apply(
    lambda x: topic_output_format_validator(x[0]['generated_text'], llama_2_output_structure_confirmation_template) if type(x)==list else ["UnIdentified"]
)
test['topic_list'] = test['topic_list'].apply(lambda x: ast.literal_eval(x.split("\n\n")[1].strip()) if "\n\n" in x else ast.literal_eval(x))

db.write_df_to_azure(
    abfsClient,
    input_file=test,
    azure_path=r'datascience/data/ds/sandbox/shibushaun/silver/final/llama_inference_prod_Jan2024.csv',
    format="csv",
    verbose=True
)

test = test[~(test['llama_inference']=={'topics_discussed': '', 'actions': '', 'summary': '', 'satisfaction_dict': ''})]
test['topics_discussed'] = test['llama_inference'].apply(lambda x: x['topics_discussed'])
test['topic_list'] = test['topics_discussed'].apply(
    lambda x: topic_output_format_validator(x[0]['generated_text'], llama_2_output_structure_confirmation_template) if type(x)==list else ["UnIdentified"]
)
test['topic_list'] = test['topic_list'].apply(lambda x: ast.literal_eval(x.split("\n\n")[1].strip()) if "\n\n" in x else ast.literal_eval(x))
test

# COMMAND ----------

# Function to count topics
def count_topics(data):
    topic_counts = {}
    for entry in data:
        for topic in entry:
            if topic in topic_counts:
                topic_counts[topic] += 1
            else:
                topic_counts[topic] = 1
    return topic_counts
    
topic_counts = test['topic_list'].value_counts()
# Assuming topic_counts contains the Counter dictionary you provided
#topic_counts_cleaned = Counter({re.sub(r'[^\w\s]', '', key): value for key, value in topic_counts.items()})

print(topic_counts)

# Count topics
topic_counts = count_topics(test['topic_list'])

# Print topic counts
for topic, count in topic_counts.items():
    print(f"{topic}: {count}")

# COMMAND ----------

ap = AudioProcessor(abfs_client = abfsClient, pytest_flag=False, db=db)

test['redacted_summary'] = test['summary'].apply(lambda x: ap.double_redact(ap.redact(x)) if x is not None else "") 
test['redacted_satisfaction_dict'] = test['satisfaction_dict'].apply(lambda x: ap.double_redact(ap.redact(x)) if x is not None else "")
test['redacted_actions'] = test['actions'].apply(lambda x: ap.double_redact(ap.redact(x)) if x is not None else "")

db.write_df_to_azure(
    abfsClient,
    input_file=test,
    azure_path=r'datascience/data/ds/sandbox/shibushaun/silver/final/llama_inference_prod_Jan2024.csv',
    format="csv",
    verbose=True
)


# COMMAND ----------

# MAGIC %md
# MAGIC **LLAMA EXECUTION TIME BENCHMARKING**

# COMMAND ----------

# MAGIC %md
# MAGIC USING LLAMA FROM INFERENCE END POINT 

# COMMAND ----------

transcription = test.at[55,'Transcription']

# COMMAND ----------

start = time.time()
summ = summary(transcription,
                summary_prompt_template)
print(time.time()-start)
print(summ)

# COMMAND ----------

start = time.time()
satisfaction_dict = satisfaction(transcription,
                                satisfaction_score_prompt_template)
print(time.time()-start)
print(satisfaction_dict)

# COMMAND ----------

start = time.time()
top = topics(summ,
            topic_inference_prompt_template)
print(time.time()-start)
print(top)

# COMMAND ----------

start = time.time()
loa = list_of_actions(transcription,
                    actions_taken_prompt_template)
print(time.time()-start)
print(loa)

# COMMAND ----------

# MAGIC %md
# MAGIC LOADING LLAMA 2 GGUF FROM HUGGINGFACE

# COMMAND ----------

pipe = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q3_K_S.gguf", model_type="llama", gpu_layers=0)
start = time.time()
print(pipe(summary_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------

start = time.time()
print(pipe(actions_taken_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------

start = time.time()
print(pipe(satisfaction_score_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------

start = time.time()
print(pipe(topic_inference_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC LOAD LLAMA 2 FROM HUGGINGFACE

# COMMAND ----------

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")
start = time.time()
print(pipe(summary_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------

start = time.time()
print(pipe(summary_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------

start = time.time()
print(pipe(actions_taken_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------

start = time.time()
print(pipe(satisfaction_score_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------

start = time.time()
print(pipe(topic_inference_prompt_template.format(transcription)))
print(time.time()-start)

# COMMAND ----------

del pipe

# COMMAND ----------

# MAGIC %md
# MAGIC LOAD LLAMA 2 LOCALLY

# COMMAND ----------

db = db_utils.DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()
blob_directory = "datascience/data/ds/sandbox/shibushaun/huggingface_models/llama_2_7b_chat"
with tempfile.TemporaryDirectory() as temp_dir:
    # List all blobs in the directory and download them
    blob_list = abfsClient.ls(blob_directory)
    for blob in blob_list:
        blob_name = os.path.basename(
            blob)  # Extract just the file name
        blob_path = os.path.join(blob_directory, blob_name)
        local_file_path = os.path.join(temp_dir, blob_name)

        with abfsClient.open(blob_path, "rb") as remote_file:
            with open(local_file_path, "wb") as local_file:
                local_file.write(remote_file.read())

    start = time.time()
    pipe = pipeline("text-generation", 
                    model=temp_dir)
    #model = AutoModelForCausalLM.from_pretrained(save_directory, device_map = 'auto')
    #tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print(time.time()-start)

# COMMAND ----------

trans = test.at[55,'Transcription']
print(trans)

# COMMAND ----------

start = time.time()
topics = pipe(topic_inference_prompt_template.format(trans))[0]['generated_text'].split("[/INST]")[1].split("\n\n")[2]
print(topics)
print(time.time()-start)

# COMMAND ----------

start = time.time()
satisfaction_score = pipe(satisfaction_score_prompt_template.format(trans))[0]['generated_text'].split("\n\n")[1]
print(satisfaction_score)
print(time.time()-start)

# COMMAND ----------

start = time.time()
actions_taken = pipe(actions_taken_prompt_template.format(trans))[0]['generated_text'].split("[/INST]\n\n")[1]
print(actions_taken)
print(time.time()-start)

# COMMAND ----------

start = time.time()
summary = pipe(summary_prompt_template.format(trans))[0]['generated_text'].split("[/INST]")[1]
print(summary)
print(time.time()-start)

# COMMAND ----------

del pipe

# COMMAND ----------

test["llama_summary"] = test["Transcription"].apply(lambda trans: pipe(summary_prompt_template.format(trans))[0]['generated_text'].split("[/INST]")[1])
test["llama_topics"] = test["Transcription"].apply(lambda trans: pipe(topic_inference_prompt_template.format(trans))[0]['generated_text'].split("[/INST]")[1].split("\n\n")[2])
test["llama_action_taken"] = test["Transcription"].apply(lambda trans: pipe(actions_taken_prompt_template.format(trans))[0]['generated_text'].split("[/INST]\n\n")[1])
test["llama_action_taken"] = test["Transcription"].apply(lambda trans: pipe(satisfaction_score_prompt_template.format(trans))[0]['generated_text'].split("\n\n")[1])

# COMMAND ----------

import concurrent.futures

def apply_functions(trans):
    if len(trans.split()) >= 50:
        summary = pipe(summary_prompt_template.format(trans))#[0]['generated_text'].split("[/INST]")[1]
        topics = pipe(topic_inference_prompt_template.format(trans))#[0]['generated_text'].split("[/INST]")[1].split("\n\n")[2]
        action_taken = pipe(actions_taken_prompt_template.format(trans))#[0]['generated_text'].split("[/INST]\n\n")[1]
        satisfaction_score = pipe(satisfaction_score_prompt_template.format(trans))#[0]['generated_text'].split("\n\n")[1]
        return [summary, topics, action_taken, satisfaction_score]
    else:
        return "", "", "", ""

def process_transcription(trans):
    return apply_functions(trans)

start = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_transcription, test["Transcription"]))

test["llama_summary"], test["llama_topics"], test["llama_action_taken"], test["llama_satisfaction_score"] = zip(*results)
print("Llama Inference Time: "+str(time.time()-start))


# COMMAND ----------

pii_masking_template = """
[INST] <<SYS>> Redact any Personally Identifiable Information (PII) found in the conversation. PII includes but is not limited to: 
- Names
- Addresses
- Phone Numbers
- Email Addresses
- Social Security Numbers
- Account Numbers
- Driver's License Numbers
- IP Addresses
- Bill Account Information
- Passport Numbers
- Bank Account Information 

Replace any PII with generic placeholders or redact them completely to ensure privacy and confidentiality. 
<<SYS>> {} [/INST]
"""

pipe(summary_prompt_template.format(trans))


# COMMAND ----------

import concurrent.futures

def apply_functions(transcriptions):
    results = []
    for trans in transcriptions:
        if len(trans.split()) >= 50:
            summary = pipe(summary_prompt_template.format(trans))[0]['generated_text'].split("[/INST]")[1]
            topics = pipe(topic_inference_prompt_template.format(trans))[0]['generated_text'].split("[/INST]")[1].split("\n\n")[2]
            action_taken = pipe(actions_taken_prompt_template.format(trans))[0]['generated_text'].split("[/INST]\n\n")[1]
            satisfaction_score = pipe(satisfaction_score_prompt_template.format(trans))[0]['generated_text'].split("\n\n")[1]
            results.append((summary, topics, action_taken, satisfaction_score))
        else:
            results.append(("", "", "", ""))
    return results

def process_transcriptions(transcriptions):
    return apply_functions(transcriptions)

batch_size = 10  # Define the batch size
results = []

# Divide transcriptions into batches
transcription_batches = [test["Transcription"][i:i+batch_size] for i in range(0, len(test["Transcription"]), batch_size)]

with concurrent.futures.ThreadPoolExecutor() as executor:
    # Process each batch of transcriptions concurrently
    for batch_results in executor.map(process_transcriptions, transcription_batches):
        results.extend(batch_results)

# Unpack the results and assign them to the respective DataFrame columns
test["llama_summary"], test["llama_topics"], test["llama_action_taken"], test["llama_satisfaction_score"] = zip(*results)


# COMMAND ----------



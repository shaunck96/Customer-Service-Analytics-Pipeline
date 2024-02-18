# Databricks notebook source
# MAGIC %md
# MAGIC **LM Inferencing**

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
                 db="",
                 whiper_model_size = 'tiny.en'):
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
        self.transcription_model = WhisperModel(whiper_model_size, #tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, distil-medium.en 	
                                                device="cpu",
                                                compute_type="int8", #int8, fp32
                                                cpu_threads=8,
                                                num_workers=4) 
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

    def redact_sequence_patterns(self, text: str) -> str:
        pattern = r'\b\w+(-\w+)+\b'
        redacted_text = re.sub(pattern, '***', text)
        return redacted_text    

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
                                red_text = self.redact_sequence_patterns(
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
                                red_text = self.redact_sequence_patterns(
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

class AllContextWindowSummaryGenerator:

    def __init__(self,
                 db="",
                 abfs_client='ABFS',
                 pytest_flag=False):
        """
        Initialize the summary generator.

        Args:
            llm (str): The language model to use (e.g.,
            'gpt-3.5-turbo', 'facebook/bart-large-cnn', etc.)
        """
        self.abfs_client = abfs_client
        self.pytest_flag = pytest_flag
        self.db = db
        if self.pytest_flag is False:
            self.config_file_path = (r"datascience/data/ds"
                                     r"/sandbox/shibushaun"
                                     r"/summarization_credentials"
                                     r"/summ_cred_new.yaml")

            with self.abfs_client.open(self.config_file_path, 'r') as file:
                config = yaml.safe_load(file)
                self.input_path = config["input_path"]
            self.call_transcriptions = self.db.read_df_from_azure(
                self.abfs_client,
                self.input_path,
                format="csv",
                verbose=True)
            path = (r"datascience/data/ds"
                    r"/sandbox/shibushaun"
                    r"/huggingface_models/philschmid/bart")
            self.sum_pipe = DBUtilConnectionCreator(  # noqa: F405
                self.db).download_and_load_bart_pipeline(self.abfs_client,
                                                         path)
            self.s_t = DBUtilConnectionCreator(  # noqa: F405
                self.db).download_and_load_bart_tokenizer(self.abfs_client,
                                                          path)
        else:
            self.config_file_path = (r"C:\Users\307164"
                                     r"\Desktop\deployment_for_bala"
                                     r"\deployment_refactored"
                                     r"\cs_pa_nlp\credentials"
                                     r"\summ_cred.yaml")

            with open(self.config_file_path, 'r') as file:
                config = yaml.safe_load(file)
                self.input_path = config['input_path']
            self.call_transcriptions = pd.read_csv(self.input_path)
            huggingface_hub.login(config['hf_token'])
            self.sum_pipe = pipeline(
                "summarization",
                model="philschmid/bart-large-cnn-samsum")
            self.s_t = AutoTokenizer.from_pretrained(
                "philschmid/bart-large-cnn-samsum")
        self.transcription_column = 'seg_tra'
        self.output_storage_path = config['output_storage_path']
        self.temp_log_file = ""
        self.logger = self.setup_logger()

    def setup_logger(self):
        """
        Set up a logger for the AllContextWindowSummaryGenerator class.

        This method creates a logger
        instance named 'AllContextWindowSummaryGenerator'
        and configures it to log messages at
        the DEBUG level. It also creates a
        temporary log file to store the log messages.

        Returns:
            logging.Logger: The configured logger instance.

        Note:
            This method should be called to initialize logging
            for the AllContextWindowSummaryGenerator class.

        Example usage:
            generator = AllContextWindowSummaryGenerator()
            logger = generator.setup_logger()
            logger.debug("This is a debug message")

        """
        logger = logging.getLogger('AllContextWindowSummaryGenerator')
        logger.setLevel(logging.DEBUG)
        self.temp_log_file = tempfile.NamedTemporaryFile(
            delete=False)
        handler = logging.FileHandler(self.temp_log_file.name)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def write_log_to_azure(self):
        """
        Write the temporary log data to an Azure Blob
        Storage container and clean up resources.

        This method flushes and seeks the temporary
        log file, generates a unique log file
        name based on the current timestamp,
        and saves the log data to a specified
        directory in an Azure Blob Storage
        container using the provided ABFS client.
        After saving, it prints a success message
        and cleans up the temporary log file.

        Note:
            Ensure that the `self.temp_log_file`
            contains the log data to be saved
            before calling this method.

        Example usage:
            generator = AllContextWindowSummaryGenerator()
            # ... Log some messages ...
            generator.write_log_to_azure()
        """
        self.temp_log_file.flush()
        self.temp_log_file.seek(0)
        log_file_name = 'summary_generator_log' + datetime.datetime.now(

        ).strftime("%Y-%m-%d_%H-%M-%S") + '.log'

        if self.db == "":
            path_to_log_file = (r"C:\Users\307164"
                                r"\Desktop"
                                r"\customer_service_insights_v2.0\logs")
            with open(
                os.path.join(
                    path_to_log_file,
                    log_file_name), 'wb') as log_file:
                log_file.write(self.temp_log_file.read())
        else:
            path_to_log_file = (r"C:\Users\307164"
                                r"\Desktop\deployment_for_bala"
                                r"\deployment_refactored\cs_pa_nlp\logs")
            DBUtilConnectionCreator(  # noqa: F405
                self.db).create_text_file(
                    self.abfs_client,
                    path_to_log_file,
                    log_file_name, "")

            with self.abfs_client.open(
                os.path.join(
                    path_to_log_file,
                    log_file_name),
                    'wb') as log_file:
                log_file.write(self.temp_log_file.read())

        print("Logs written to "+path_to_log_file+" successfully")

        self.temp_log_file.close()
        os.unlink(self.temp_log_file.name)

    def preprocess(self, transcriptions):
        merged_transcription = ""
        current_speaker = 1

        for i, transcript in enumerate(ast.literal_eval(transcriptions)):
            text = transcript['text'].strip()
            merged_transcription += f"Speaker {current_speaker}: \"{text}\"\n"

            current_speaker = 1 if current_speaker == 2 else 2

        return (merged_transcription)

    def summary_generator(self, input_text: dict) -> str:
        """
        Generate a summary for the given
        text using the specified language model.

        Args:
            text (str): The input text to summarize.

        Returns:
            list: A list containing a flag
            (1 for success, 0 for failure) and the summary text.
        """
        input_text = self.preprocess(input_text)

        try:
            if len(self.s_t.tokenize(input_text)) > 1000:
                chunk_length = 1000
                chunks = []
                current_chunk = ""

                for line in sent_tokenize(input_text):
                    encoded_line = self.s_t.encode(line,
                                                   add_special_tokens=False)

                    if len(current_chunk) + len(encoded_line) < chunk_length:
                        current_chunk += line + '\n'
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = line + '\n'

                if current_chunk:
                    chunks.append(current_chunk.strip())

                final_summary = ""
                for chunk in chunks:
                    summary = self.sum_pipe(chunk)
                    final_summary += summary[0]['summary_text'] + ' '

                return (final_summary)

            elif len(self.s_t.tokenize(input_text)) == 0:
                return ("Summary not generated for this transcription")

            else:
                final_summary = self.sum_pipe(input_text)[0]['summary_text']
                return (final_summary)

        except Exception as e:
            self.logger.error(f'Error in summary_generator: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in summary_generator: {str(e)}")
        return (summary)

    def column_of_text_summary_generator(self):
        """
        Generate summaries for each text in a DataFrame column.
        """
        self.call_transcriptions.drop_duplicates(
            [self.transcription_column], inplace=True)
        self.call_transcriptions[self.transcription_column] = \
            self.call_transcriptions[self.transcription_column].astype(str)
        print("Summarization has started: .......")
        try:
            self.call_transcriptions['Summary'] = \
                self.call_transcriptions[self.transcription_column].apply(
                    lambda x: self.summary_generator(
                        x))
            if self.pytest_flag is False:
                self.db.write_df_to_azure(self.abfs_client,
                                          input_file=self.call_transcriptions,
                                          azure_path=self.output_storage_path,
                                          format="csv",
                                          verbose=True)
            else:
                self.call_transcriptions.to_csv(self.output_storage_path)
        except Exception as e:
            self.logger.error(f'Error in \
                              column_of_text_summary_generator: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in column_of_text_summary_generator: {str(e)}")

    def summary_generation_trigger(self):
        """
        Trigger the summary generation process.
        """
        self.column_of_text_summary_generator()


class TopicGenerator:

    def __init__(self,
                 db="",
                 abfs_client='ABFS',
                 pytest_flag=False):
        """
        Initialize the topic generator.

        Args:

        """

        self.ci = {
            "Incorrect Charges": ["erroneous charges", "disputes", "unauthorized fees", "pricing discrepancies", "I didn't authorize this charge", "My bill is higher than I expected"],  # noqa: E501
            "High Bills": ["unexpectedly high bills", "excessive charges", "abnormally expensive invoices", "Why is my bill so high?", "I can't afford this bill"],  # noqa: E501
            "Payment Processing": ["failed transactions", "payment errors", "processing issues", "My payment didn't go through", "Why isn't my payment processing?"],  # noqa: E501
            "Meter Readings": ["meter reading discrepancies", "inaccuracy", "misinterpretations", "There's a mistake in my meter reading", "Why is my meter reading so high?"],  # noqa: E501
            "Auto-Pay": ["setting up auto-pay", "managing automatic payments", "failures", "cancellations", "I need help setting up auto-pay", "My auto-payment failed"],  # noqa: E501
            "Refund Requests": ["refund inquiries", "overpayments", "refund eligibility", "Can I get a refund?", "I accidentally overpaid my bill"],  # noqa: E501
            "Billing Frequency": ["billing cycle changes", "interval adjustments", "frequency updates", "Why did my billing cycle change?", "Can I change my billing frequency?"],  # noqa: E501
            "Paperless Billing": ["opting for paperless billing", "digital statements", "electronic invoices", "How do I opt for paperless billing?", "I can't access my digital statement"],  # noqa: E501
            "Bill Due Date": ["due dates", "extensions", "late payment deadlines", "payment timing", "Can I extend my bill due date?", "What happens if I miss my payment deadline?"],  # noqa: E501
            "Billing History": ["accessing past billing history", "previous invoices", "How can I view my billing history?", "Can I get a copy of a previous invoice?"],  # noqa: E501
            "Third-party Payments": ["paying through third-party processors", "transaction delays", "I paid through PayPal but my bill hasn't been updated", "My payment is taking longer than usual to process"],  # noqa: E501
            "Payment options": ["available payment methods", "payment choices", "payment selection", "What payment options do you accept?", "Can I pay with a check?"],  # noqa: E501
            "Late payment fees": ["late payment penalties", "overdue charges", "fines", "Will I be charged a late payment fee?", "How much is the overdue charge?"],  # noqa: E501
            "Payment extensions": ["extending payment dates", "modifying payment plans", "adjusting terms", "Can I extend my payment date?", "How can I modify my payment plan?"],  # noqa: E501
            "Paperless billing": ["paperless billing issues", "electronic statements", "digital invoices", "I'm not receiving my digital statement", "How do I view my digital invoice?"],  # noqa: E501
            "Payment disputes": ["billing disputes", "payment conflicts", "contract disagreements", "I disagree with my bill", "I don't agree with the contract terms"],  # noqa: E501
            "Payment history": ["accessing payment history", "transaction history", "past invoices", "Can I see my payment history?", "How can I view my past invoices?"],  # noqa: E501
            "Online payment": ["making payments online", "digital payment methods", "online transactions", "I'm having trouble making a payment online", "Can I make a payment over the phone?"],  # noqa: E501
            "Credit card payment": ["payment inquiries with credit cards", "credit card use", "credit card transactions", "Why was my credit card declined?", "Can I split my payment between two credit cards?"],  # noqa: E501
            "Payment processing": ["payment processing concerns", "handling issues", "transaction difficulties", "My payment is stuck in processing", "I received a transaction error message"],  # noqa: E501
            "Payment receipts": ["confirmation of payment receipts", "payment validation", "How do I know my payment went through?", "Can I get a receipt for my payment?"],  # noqa: E501
            "Payment confirmation": ["verifying payment", "confirming payment", "Did you receive my payment?", "Why haven't I received a confirmation of payment?"],  # noqa: E501
            "Payment due date": ["payment due dates", "extensions", "timing", "When is my payment due?", "Can I change my payment due date?"],  # noqa: E501
            "Outage notifications": ["receiving outage alerts", "service disruption notifications", "warnings", "I didn't receive an outage notification", "How do I sign up for outage alerts?"],  # noqa: E501
            "Reporting an outage": ["reporting service disruptions", "outage procedures", "sudden interruptions", "I have an outage, what do I do?", "How do I report an outage?"],  # noqa: E501
            "Estimated restoration time": ["expected restoration times", "service recovery estimates", "When will my power be back on?", "How long will the service be disrupted?"],  # noqa: E501
            "Frequent outages": ["recurring service disruptions", "frequent outages", "Why do I experience frequent outages?", "What is the cause of the disruption?"],  # noqa: E501
            "Surge protection": ["power surge protection", "surge suppressors", "voltage stability", "How can I protect against power surges?", "Do I need a surge protector?"],  # noqa: E501
            "Emergency power": ["backup power options", "emergency power during outages", "generators", "Do you offer backup power?", "What are my options for backup power?"],  # noqa: E501
            "Power outage causes": ["reasons behind power outages", "disruption causes", "What causes power outages?", "Is there a common reason for outages?"],  # noqa: E501
            "Weather-related outages": ["outages due to adverse weather conditions", "storm-related disruptions", "Why do I experience service disruptions during storms?", "How can I prepare for weather-related outages?"],  # noqa: E501
            "Interpreter services": ["interpreter services", "language translation support", "Do you offer language support?", "Can I speak to someone in Spanish?"],  # noqa: E501
            "Payment plan options": ["payment plans", "installment choices", "payment arrangements", "Are payment plans available?", "Can I set up a payment arrangement?"],  # noqa: E501
            "Financial assistance": ["payment assistance", "financial support", "expense management", "Do you offer financial assistance?", "What type of financial support is available?"],  # noqa: E501
            "Low-income payment programs": ["programs for low-income customers", "household assistance", "individual support", "Are there payment options for low-income households?", "What programs are available for individuals in need?"],  # noqa: E501
            "Debt relief programs": ["programs for managing outstanding debt", "debt relief assistance", "How can I manage my outstanding debt?", "What is the debt relief program?"],  # noqa: E501
            "Budget billing": ["budget-friendly billing options", "payment stability", "equalized plans", "What are budget billing options?", "How can I stabilize my payments?"],  # noqa: E501
            "Payment assistance application": ["applying for payment assistance", "assistance application process", "Can I apply for financial assistance?", "How do I apply for financial support?"],  # noqa: E501
            "Income verification": ["verifying income", "income validation requirements", "What type of income verification is needed?", "How do I provide proof of income?"],  # noqa: E501
            "Assistance eligibility": ["eligibility criteria", "qualification requirements", "What are the eligibility requirements for assistance?", "What are the qualifications for financial support?"],  # noqa: E501
            "Payment assistance agencies": ["organizations providing financial assistance", "payment support", "What organizations provide assistance?", "Where can I find assistance?"],  # noqa: E501
            "Energy assistance programs": ["programs for energy bill assistance", "energy expense support", "Are there programs for energy bill assistance?", "What type of support is available for energy expenses?"],  # noqa: E501
            "New service setup": ["initiating new utility service", "activating new accounts", "How do I start a new utility service?", "What is the process for activating a new account?"],  # noqa: E501
            "Account activation": ["activating utility accounts", "new account setup", "starting new services", "How do I activate my utility account?", "What is involved in setting up a new account?"],  # noqa: E501
            "Transfer of service": ["transferring utility service", "changing service providers", "Can I transfer my utility service?", "How do I change my service provider?"],  # noqa: E501
            "Connection fees": ["service activation fees", "connection expenses", "Are there fees for activating service?", "What is the cost for connection expenses?"],  # noqa: E501
            "Switching service providers": ["transitioning to new utility companies", "changing providers", "Can I switch to a new utility company?", "How do I change my provider?"],  # noqa: E501
            "Contract termination": ["ending utility contracts", "contract cancellation", "termination process", "How do I end my utility contract?", "What is the termination process?"],  # noqa: E501
            "Early termination fees": ["fees for canceling contracts early", "early termination penalties", "Are there fees for canceling my contract early?", "What are the penalties for early termination?"],  # noqa: E501
            "Comparing utility rates": ["utility rate comparison", "rate plan analysis", "energy price evaluation", "How do I compare utility rates?", "What's the best rate plan for me?"],  # noqa: E501
            "Service transition": ["transitioning between utility services", "changing service types", "Can I change my service type?", "How do I transition between utility services?"],  # noqa: E501
            "Porting phone numbers": ["transferring phone numbers", "number portability", "phone number change process", "How do I transfer my phone number?", "Can I keep my current phone number?"],  # noqa: E501
            "Service installation": ["installing utility service", "service setup", "activation process", "What is the utility service installation process?", "How do I set up my service?"],  # noqa: E501
            "Service relocation": ["moving utility service", "transferring service", "changing service location", "How do I move my utility service?", "Can I transfer my service to a different location?"],  # noqa: E501
            "Service contracts": ["utility service agreements", "contract terms", "agreement details", "What is the utility service agreement?", "What are the contract terms?"],  # noqa: E501
            "On Track": ["On Track program", "customer assistance benefits", "On track program advantages", "What is the On Track program?", "What are the benefits of the On Track program?"],  # noqa: E501
            "Renewable energy initiatives": ["renewable energy efforts", "sustainability initiatives", "environmental impact", "What is the company's stance on renewable energy?", "What initiatives are being undertaken to promote sustainability?"],  # noqa: E501
            "Rebate and incentive programs": ["energy efficiency incentives", "rebate programs", "savings opportunities", "Are there incentive programs for energy efficiency?", "How can I take advantage of rebate opportunities?"],  # noqa: E501
            "Energy efficiency tips": ["energy conservation tips", "energy-saving strategies", "efficiency recommendations", "What are some ways to conserve energy?", "What are the best energy-saving strategies?"],  # noqa: E501
            "Green energy options": ["environmentally friendly energy sources", "clean energy alternatives", "What are some green energy options?", "Are there clean energy alternatives available?"],  # noqa: E501
            "Start Service": ["initiating utility service", "new account activation", "beginning service", "How do I initiate utility service?", "What is involved in new account activation?"],  # noqa: E501
            "Stop Service": ["terminating utility service", "canceling service", "ending service", "How do I terminate my utility service?", "What is the process for canceling service?"],  # noqa: E501
            "Appliance upgrade programs": ["programs for upgrading appliances", "energy-efficient options", "Are there programs for upgrading to energy-efficient appliances?", "What are my options for energy-efficient appliances?"],  # noqa: E501
            "Smart Meter Installation": ["smart meter setup", "digital meter installation", "How do I install a smart meter?", "What are the advantages of a digital meter?"],  # noqa: E501
            "Water Quality Issues": ["water taste problems", "discolored water", "water odor", "Why does my tap water taste strange?", "How can I improve my water quality?"],  # noqa: E501
            "Water Pressure Problems": ["low water pressure", "no water flow", "weak water pressure", "My shower has low water pressure", "How can I fix my water pressure issue?"],  # noqa: E501
            "Sewer Blockages": ["clogged sewer", "blocked drain", "sewage backup", "My toilet is clogged", "What to do if my sewer is blocked?"],  # noqa: E501
            "Meter Reading Access": ["accessing my meter readings", "meter reading information", "Where can I find my meter reading?", "How often should I check my meter?"],  # noqa: E501
            "Solar Panel Queries": ["solar power installation", "solar panel questions", "solar energy benefits", "Can I install solar panels?", "How do solar panels work?"],  # noqa: E501
            "Green Energy Subscriptions": ["renewable energy subscription", "green power options", "eco-friendly plans", "Can I subscribe to green energy?", "What are the benefits of green energy?"],  # noqa: E501
            "Gas Odor Complaints": ["gas smell in my home", "natural gas odor", "emergency gas leak", "I smell gas in my house", "What should I do in case of a gas leak?"],  # noqa: E501
            "Voltage Fluctuations": ["electricity voltage problems", "fluctuating voltage", "power surges", "Why do my lights flicker?", "How can I protect my devices from power surges?"],  # noqa: E501
            "Water Heater Issues": ["water heater problems", "no hot water", "leaking water heater", "My water heater isn't working", "How to fix a leaking water heater?"],  # noqa: E501
            "Home Energy Audit": ["energy efficiency assessment", "home energy inspection", "Can I get an energy audit?", "How can I make my home more energy-efficient?"],  # noqa: E501
            "Gas Appliance Safety": ["gas appliance maintenance", "gas appliance safety tips", "Are my gas appliances safe?", "How often should I service my gas appliances?"],  # noqa: E501
            "Environmental Concerns": ["utility company's environmental initiatives", "carbon footprint reduction", "What is the company doing for the environment?", "How can I reduce my environmental impact?"],  # noqa: E501
            "New Utility Technologies": ["emerging utility technologies", "future of utilities", "What's the future of utilities?", "How can I stay updated on new utility technologies?"],  # noqa: E501
            "Emergency Preparedness": ["emergency preparedness tips", "disaster readiness", "What should I have in my emergency kit?", "How can I prepare for power outages?"],  # noqa: E501
            "Community Outreach Programs": ["utility company community programs", "charity partnerships", "How can I get involved in community programs?", "What charities does the company support?"],  # noqa: E501
            "Billing Address Change": ["updating billing address", "change of residence", "I'm moving, how do I update my billing address?", "Can I change my service location?"],  # noqa: E501
            "Appliance Repair Services": ["appliance repair requests", "broken appliances", "My appliance is not working", "How can I request an appliance repair?"],  # noqa: E501
            "Customer Feedback and Surveys": ["customer satisfaction survey", "feedback opportunities", "Can I provide feedback about your service?", "How can I participate in customer surveys?"],  # noqa: E501
            "Renewable Energy Certificates": ["RECs purchase", "buying renewable energy certificates", "How can I buy RECs?", "What is the benefit of purchasing RECs?"],  # noqa: E501
            "Home Battery Storage": ["home battery solutions", "battery backup installation", "Can I install a home battery?", "How does home battery storage work?"],  # noqa: E501
            "Energy Usage Analysis": ["energy consumption analysis", "usage tracking tools", "How can I track my energy usage?", "What are the tools for energy consumption analysis?"],  # noqa: E501
            "Smart Home Integration": ["smart home utilities", "home automation systems", "Can I integrate my home with smart utilities?", "What are the benefits of a smart home?"],  # noqa: E501
            "Data Privacy Concerns": ["customer data security", "privacy policies", "How is my data protected?", "What are your privacy practices?"],  # noqa: E501
            "Sustainable Transportation": ["electric vehicle charging", "EV charging stations", "Where can I charge my electric vehicle?", "What's the cost of charging an EV?"],  # noqa: E501
            "Power Line Issues": ["hanging power lines", "damaged power lines", "power line safety", "What should I do if I see a hanging power line?", "How to report a power line issue?,power outages due to damaged power lines", "line-related disruptions", "Why is there a power outage caused by power lines?", "How long will it take to restore power in case of a line-related outage?"],  # noqa: E501
            "Street Light Outages": ["non-functional street lights", "dark streets", "reporting broken street lights", "How to report a street light that's out?", "When will the street light be repaired?"],  # noqa: E501
            "Tree related Outage": ["trees fallen on power lines", "damage caused by uprooted trees", "What to do if an uprooted tree affects power lines?", "How quickly can the uprooted tree be removed and power restored?"],  # noqa: E501
            "Power Restoration Schedule": ["scheduling power restoration", "estimated restoration times", "When will my power be turned back on?", "How can I get a schedule for power restoration?"]  # noqa: E501
        }

        self.pytest_flag = pytest_flag
        self.abfs_client = abfs_client
        self.db = db
        self.config_file_path = (r'datascience/data/ds'
                                    r'/sandbox/shibushaun'
                                    r'/topic_modelling_credentials'
                                    r'/topic_cred_new.json')
        with self.abfs_client.open(self.config_file_path, 'r') as file:
            self.config = json.load(file)
        csv_file_path = self.config['csv_file_path']
        with self.abfs_client.open(csv_file_path, 'r') as file:
            self.call_transcriptions = final_df
        path = (r"datascience/data/ds/sandbox"
                r"/shibushaun/huggingface_models"
                r"/bart_zeroshot")
        self.classifier = DBUtilConnectionCreator(  # noqa: F405
            self.db).download_and_load_zeroshot_model(
                self.abfs_client,
                path)
        self.call_transcriptions_column = 'Summary'
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.output_storage_path = self.config['output_storage_path']
        self.temp_log_file = ""
        self.logger = self.setup_logger()

    def setup_logger(self):
        """
        Set up a logger for the TopicGenerator class.

        This method creates a logger instance named
        'TopicGenerator' and configures it to log messages at the DEBUG level.
        It also creates a temporary log file to store the log messages.

        Returns:
            logging.Logger: The configured logger instance.

        Note:
            This method should be called to initialize
            logging for the TopicGenerator class.

        Example usage:
            generator = TopicGenerator()
            logger = generator.setup_logger()
            logger.debug("This is a debug message")

        """
        logger = logging.getLogger('TopicGenerator')
        logger.setLevel(logging.DEBUG)
        self.temp_log_file = tempfile.NamedTemporaryFile(
            delete=False)
        handler = logging.FileHandler(self.temp_log_file.name)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def write_log_to_azure(self):
        """
        Write the temporary log data to an
        Azure Blob Storage container and clean up resources.

        This method flushes and seeks the temporary log file,
        generates a unique log file name based on the current timestamp,
        and saves the log data to a specified directory
        in an Azure Blob Storage container using the provided ABFS client.
        After saving, it prints a success message
        and cleans up the temporary log file.

        Note:
            Ensure that the `self.temp_log_file`
            contains the log data to be saved
            before calling this method.

        Example usage:
            generator = TopicGenerator()
            # ... Log some messages ...
            generator.write_log_to_azure()
        """
        self.temp_log_file.flush()
        self.temp_log_file.seek(0)
        lfn = 'topic_modeller_log' + datetime.datetime.now(

        ).strftime("%Y-%m-%d_%H-%M-%S") + '.log'

        if self.db == "":
            ptlf = (r"C:\Users\307164"
                    r"\Desktop\deployment_for_bala"
                    r"\deployment_refactored"
                    r"\cs_pa_nlp\logs")
            with open(os.path.join(ptlf,
                                   lfn), 'wb') as log_file:
                log_file.write(self.temp_log_file.read())
        else:
            ptlf = "datascience/data/ds/sandbox/shibushaun/logs"
            DBUtilConnectionCreator(  # noqa: F405
                self.db).create_text_file(
                    self.abfs_client,
                    ptlf,
                    lfn,
                    "")

            with self.abfs_client.open(os.path.join(ptlf,
                                                    lfn), 'wb') as log_file:
                log_file.write(self.temp_log_file.read())

        print("Logs written to "+ptlf+" successfully")

        self.temp_log_file.close()
        os.unlink(self.temp_log_file.name)

    def extract_summary(self, input_string):
        """
        Extract a summary from the input string by removing special characters.

        This method takes an input string and removes any
        characters that are not alphanumeric, whitespace, or period ('.').
        The cleaned string is returned as the summary.

        Args:
            input_string (str): The input string from
            which to extract the summary.

        Returns:
            str: The cleaned summary string.

        Raises:
            Exception: If an error occurs during the extraction process,
              it is logged, and an error message is printed.

        Example usage:
            generator = TopicGenerator()
            input_text = "This is a sample text
            with special characters @ and $."
            cleaned_summary = generator.extract_summary(input_text)
            print(cleaned_summary)
        """
        try:
            cleaned_string = re.sub(r'[^\w\s.]', '', input_string)
            return cleaned_string

        except Exception as e:
            self.logger.error(f'Error in extract_summary: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in extract_summary: {str(e)}")

    def topic_modeller(self, transcription, topic_embeddings):
        """
        Extracts and identifies topics from a given
        transcription using cosine similarity
        between sentence embeddings and predefined
        topic embeddings.

        Args:
        transcription (str): The input transcription
        to analyze for topics.
        topic_embeddings (list of numpy arrays):
          Precomputed embeddings for predefined topics.

        Returns:
        list of str: A list of identified topics based on cosine similarity.

        The function processes the input transcription
        by splitting it into sentences and computing
        embeddings for each sentence using a pre-trained model.
        Then, it calculates the cosine similarity
        between each sentence embedding and the embeddings
        of predefined topics. Sentences with cosine
        similarity above a certain threshold are
        considered to belong to those topics.

        The identified topics are ranked based on
        cosine similarity and sentence position. If more than 10
        sentences are identified for topics, the top 10 topics
        are returned; otherwise, all identified topics
        are returned. If no topics are identified,
        an empty list is returned.

        Example:
        transcription = "I have a problem with my phone.
        The battery drains quickly. It's also slow."
        topic_embeddings =
        [topic1_embedding, topic2_embedding, ...]
        # Precomputed topic embeddings
        identified_topics =
        topic_modeller(transcription, topic_embeddings)
        # Output: ['Battery Issues', 'Performance Problems']
        """
        try:
            index = 0
            threshold = 0.35
            top_2_topics_per_cluster = pd.DataFrame(
                columns=[
                    'Sentence',
                    'Topic',
                    'Position',
                    'cos_sim',
                    'Chunking Strategy'])
            print("chunks")
            chunks = list(transcription.split('.'))
            chunks = [sentence for sentence in transcription.split(
                '.') if len(sentence.split()) >= 5]
            print("sentence_embeddings")
            sentence_embeddings = self.model.encode(chunks)
            print("Dot Product Computation")
            for i, sentence_embedding in enumerate(sentence_embeddings):
                for topic_num, topic_embedding in enumerate(topic_embeddings):
                    dot_product = np.dot(sentence_embedding, topic_embedding)
                    norm_A = np.linalg.norm(sentence_embedding)
                    norm_B = np.linalg.norm(topic_embedding)
                    cosine_similarity = dot_product / (norm_A * norm_B)
                    if cosine_similarity > threshold:
                        top_2_topics_per_cluster.at[index,
                                                    'Sentence'] = str(
                                                        chunks[i])
                        top_2_topics_per_cluster.at[index,
                                                    'Topic'] = str(
                                                        list(
                                                            self.ci.keys())[
                                                                topic_num])
                        top_2_topics_per_cluster.at[index,
                                                    'Position'] = i
                        top_2_topics_per_cluster.at[index,
                                                    'cos_sim'] = float(
                                                        cosine_similarity)
                        top_2_topics_per_cluster.at[index,
                                                    'Chunking Strategy'] = str(
                                                        chunks)
                        index += 1

            if len(top_2_topics_per_cluster) == 0:
                print("Empty top topics df")
                return []

            position_wise = top_2_topics_per_cluster.sort_values(by=[
                'Position'], ascending=True)
            if len(position_wise) >= 10:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'].iloc[0:10])
                print("Top 10 Identified")
            elif len(position_wise) > 0:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'])
                print(f"Top {len(top_topics)} Identified")
            else:
                top_topics = []
                print("No Topics")
            return top_topics
        except Exception as e:
            self.logger.error(f'Error in topic_modeller: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in topic_modeller: {str(e)}")
            return []

    def probability_assignment(self, summary, topic_list):
        """
        Assigns probabilities or classification results
        to a summary based on identified topics.

        Args:
        summary (str): The summary text to assign
        probabilities or classify.
        topic_list (list of str): A list of identified
        topics for the summary.

        Returns:
        dict or str: If one topic is identified, a
        dictionary with the topic label and its probability
                    is returned. If multiple topics are
                    identified, the result from the classifier is
                    returned, typically containing labels
                      and corresponding probabilities. If no topics
                    are identified, "UNIDENTIFIED" is returned.

        This function takes a summary text and a
        list of identified topics as input.
        If no topics are
        identified, it returns "UNIDENTIFIED."
        If one topic is identified, it classifies the summary
        based on that topic and returns a
        dictionary with the topic label and its probability. If multiple
        topics are identified, it returns
        the result from the classifier, which typically includes labels
        and corresponding probabilities.

        Example:
        summary = "The battery of my phone drains quickly."
        topic_list = ['Battery Issues']
        classification_result = probability_assignment(summary, topic_list)
        # Output: {'Battery Issues': 0.85}
        """
        print("probability_assignment function started")
        try:
            if len(topic_list) == 0:
                print("Unidentified")
                return "UNIDENTIFIED"
            return (self.classifier(summary, topic_list))
        except Exception as e:
            self.logger.error(f'Error in probability_assignment: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in probability_assignment: {str(e)}")

    def apply_probability_assignment(self, row):
        """
        Apply probability assignment to a single row of data.

        Args:
        row (pandas.Series): A single row of data
        containing summary and identified topics.

        Returns:
        dict, str: If topics are identified,
        it returns a dictionary with topic labels and their
                corresponding probabilities.
                If no topics are identified, "UNIDENTIFIED" is returned.

        This function takes a single row of data
        (typically from a DataFrame) with a summary and a list
        of identified topics. It then
        applies the `probability_assignment` function to assign
        probabilities or classifications
        to the summary based on the identified topics.

        Example:
        row = {'summary_clean':
        "The battery of my phone drains quickly.",
        'Sub_Topic': ['Battery Issues']}
        result = apply_probability_assignment(row)
        # Output: {'Battery Issues': 0.85}
        """
        print("row wise probability_assignment started")
        try:
            if len(row['Sub_Topic']) == 0:
                print("Unidentified")
                return "UNIDENTIFIED"
            else:
                summary = row['summary_clean']
                topic_list = row['Sub_Topic']
                print(topic_list)
                probabilities = self.probability_assignment(
                    summary, topic_list)
                return probabilities
        except Exception as e:
            self.logger.error(f'Error in\
                               apply_probability_assignment: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in apply_probability_assignment: {str(e)}")

    def parse_topic_with_probabilities(self, x):
        """
        Parse a topic with probabilities from input data.

        Args:
        x (dict): Input data, usually a dictionary.

        Returns:
        dict: Parsed topic with probabilities.

        If the input 'x' is a dictionary,
        it is returned as is. Otherwise,
        a default dictionary with 'Unidentified'
        and probability 1 is returned.
        """
        try:
            if type(x) is dict:
                return x
        except (IndexError, ValueError, SyntaxError):
            pass
        return {'Unidentified': 1}

    def get_primary_topic(self, x):
        """
        Get the primary topic from input data.

        Args:
        x (dict): Input data, usually a dictionary containing 'labels' field.

        Returns:
        str: The primary topic label.

        If 'x' is not a valid dictionary or
        does not contain 'labels', 'Unidentified' is returned.
        """
        try:
            return x[list(x.keys())[1]][0]
        except (IndexError, TypeError):
            return 'Unidentified'

    def get_secondary_topic(self, x):
        """
        Get the secondary topic from input data.

        Args:
        x (dict): Input data, usually a dictionary containing 'labels' field.

        Returns:
        str: The secondary topic label.

        If 'x' is not a valid dictionary or
        does not contain 'labels', 'None' is returned.
        """
        try:
            if len(list(x.keys())) > 1:
                return x[list(x.keys())[1]][1]
            else:
                return 'None'
        except (IndexError, TypeError):
            return 'None'

    def topic_generator(self):
        """
        Generate topics and probabilities for call transcriptions.

        This method performs several data processing steps,
        including cleaning and topic estimation,
        and stores the results in the 'call_transcriptions' dataframe.

        Returns:
        None
        """
        print("Duplicates Dropped")
        self.call_transcriptions.drop_duplicates([
            self.call_transcriptions_column],
            inplace=True)
        print("Converted to string")
        self.call_transcriptions[
            self.call_transcriptions_column] = self.call_transcriptions[
                self.call_transcriptions_column].astype(str)
        print("Clean Summary")
        self.call_transcriptions[
            'summary_clean'] = self.call_transcriptions[
                self.call_transcriptions_column].apply(
                    lambda x: self.extract_summary(x))
        sub_topic_embeddings = self.model.encode(
            (list(self.ci.values())))

        print("Number of Rows \
            Fed Into Topic Estimation\
            Model: "+str(len(self.call_transcriptions)))
        print("Topic Estimation has begun")

        start = time.time()
        print("Sub_Topic")
        self.call_transcriptions[
            'Sub_Topic'] = self.call_transcriptions[
                'summary_clean'].apply(lambda x: self.topic_modeller(
                    x,
                    sub_topic_embeddings))
        print("topic_with_probabilities")
        self.call_transcriptions[
            'topic_with_probabilities'] = self.call_transcriptions.apply(
                self.apply_probability_assignment,
                axis=1)
        self.call_transcriptions['Primary_Topic'] = ""
        self.call_transcriptions['Secondary_Topic'] = ""

        self.call_transcriptions[
            'topic_dict'] = self.call_transcriptions[
                'topic_with_probabilities'].apply(
                    self.parse_topic_with_probabilities)
        self.call_transcriptions[
            'Primary_Topic'] = self.call_transcriptions[
                'topic_dict'].apply(
                    self.get_primary_topic)
        self.call_transcriptions[
            'Secondary_Topic'] = self.call_transcriptions[
                'topic_dict'].apply(
                    self.get_secondary_topic)
        return self.call_transcriptions        

        print("Time For Execution: "+str(time.time()-start))

class sentiment_eval:
    def __init__(self,
                 db="",
                 abfs_client='ABFS',
                 pytest_flag=False):
        """
        Initialize the sentiment evaluator.

        Args:
            db (str): Database connection information.
            abfs_client (str): Azure Blob Storage client.
            pytest_flag (bool): Flag to indicate if running in pytest mode.
        """
        self.abfs_client = abfs_client
        self.pytest_flag = pytest_flag
        self.config_file_path = (r"datascience/data/ds"
                                    r"/sandbox/shibushaun"
                                    r"/sentiment_analysis_credentials"
                                    r"/sentiment_analysis_cred_new.yaml")
        with self.abfs_client.open(self.config_file_path, 'r') as file:
            config = yaml.safe_load(file)
            self.input_path = config['input_path']
        self.db = db
        self.call_transcriptions = final_df
        path = (r"datascience/data/ds"
                r"/sandbox/shibushaun"
                r"/huggingface_models"
                r"/cardiffnlp"
                r"/twitter-roberta-base-sentiment-latest")
        self.pipe = DBUtilConnectionCreator(  # noqa: F405
            self.db).download_and_load_sentiment_model(
                self.abfs_client,
                path)

        self.transcription_column = 'SegmentedTranscription'
        self.output_storage_path = config['output_storage_path']
        self.temp_log_file = ""
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('SentimentEval')
        logger.setLevel(logging.DEBUG)
        self.temp_log_file = tempfile.NamedTemporaryFile(
            delete=False)
        handler = logging.FileHandler(self.temp_log_file.name)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def write_log_to_azure(self):
        self.temp_log_file.flush()
        self.temp_log_file.seek(0)
        log_file_name = 'sentiment_computer_log' + datetime.datetime.now(

        ).strftime("%Y-%m-%d_%H-%M-%S") + '.log'

        if self.db == "":
            path_to_log_file = (r"C:\Users\307164"
                                r"\Desktop\deployment_for_bala"
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
                    log_file_name,
                    "")

            with self.abfs_client.open(
                os.path.join(path_to_log_file, log_file_name), 'wb'
            ) as log_file:
                log_file.write(self.temp_log_file.read())

        print("Logs written to "+path_to_log_file+" successfully")

        self.temp_log_file.close()
        os.unlink(self.temp_log_file.name)

    def sentiment_scores(self, sentence):
        """
        Calculate sentiment score for a given sentence.

        Args:
            sentence (str): Input sentence.

        Returns:
            str: Sentiment label ('Positive', 'Negative', or 'Neutral').

        Uses the SentimentIntensityAnalyzer from the nltk library
        to calculate the sentiment score and classify
        it as Positive, Negative, or Neutral.
        """

        return (self.pipe(sentence)[0]['label'])

    def sentiment_computer(self, chunk):
        """
        Calculate the most frequent sentiment label for a chunk of text.

        Args:
            chunk (str): Input text chunk.

        Returns:
            str: Most frequent sentiment label
            ('Positive', 'Negative', or 'Neutral') in the chunk.

        Splits the chunk into sentences and
        calculates the sentiment for each sentence,
        then determines the most frequent sentiment
        label in the chunk.
        """
        tokens_sent = re.compile('[.!?] ').split(chunk)
        sentiment_list = []
        for sentence in tokens_sent:
            sentiment_list.append(self.sentiment_scores(sentence))
        counts = Counter(sentiment_list)
        most_frequent_sentiment = counts.most_common(1)[0][0]
        return (most_frequent_sentiment)

    def segment_sentiment_computer(self, chunk):
        """
        Calculate sentiment labels for segments within a chunk.

        Args:
            chunk (list): List of segments, each containing 'text' field.

        Returns:
            list: List of segments with 'sentiment' field added,
            indicating the sentiment label
            ('Positive', 'Negative', or 'Neutral')
            for each segment.

        Iterates through the segments in the chunk and calculates
        sentiment labels for each segment.
        """
        for segment in chunk:
            segment['sentiment'] = self.sentiment_computer(segment['text'])
        return chunk

    def sentiment_emotion_classifier(self):
        """
        Classify sentiment for call transcriptions.

        Adds a 'Sentiment' column to the 'call_transcriptions' dataframe,
        indicating the sentiment label ('Positive', 'Negative', or 'Neutral')
        for each call transcription segment.

        Writes the updated dataframe to Azure Blob Storage
        or a local CSV file based on the pytest_flag.
        """
        self.call_transcriptions[
            'Sentiment'] = self.call_transcriptions[
                'SegmentedTranscription'].apply(
                    lambda x: self.segment_sentiment_computer(
                        ast.literal_eval(
                            str(x))) if len(
                                x) > 0 else "No Sentiment Identified")
        return self.call_transcriptions



# COMMAND ----------

db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()

#ap_tiny = AudioProcessor(abfs_client = abfsClient,  pytest_flag=False, db=db,whiper_model_size = 'tiny.en')
#ap_base = AudioProcessor(abfs_client = abfsClient,pytest_flag=False, db=db, whiper_model_size = 'base.en')
#ap_medium = AudioProcessor(abfs_client = abfsClient, pytest_flag=False, db=db,whiper_model_size = 'medium.en')
#ap_large = AudioProcessor(abfs_client = abfsClient, pytest_flag=False, db=db,whiper_model_size = 'large-v1')
ap_distil_med = AudioProcessor(abfs_client = abfsClient, pytest_flag=False, db=db,whiper_model_size = 'distil-medium.en')

summ = AllContextWindowSummaryGenerator(abfs_client = abfsClient, pytest_flag=False, db=db)

# COMMAND ----------

file_name = "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_2024/6054385.wav"

with open(file_name, 'rb'):
    y, sr = librosa.load(file_name)
    duration = librosa.get_duration(y=y, sr=sr)

start = time.time()

tra = ap_distil_med.faster_transcriber(y)
transcriptions_text = "".join([t['text'] for t in tra])
redacted_text = ap_distil_med.double_redact(ap_distil_med.redact(transcriptions_text))

print("Transcription: "+str(transcriptions_text))
print("\nRedacted Transcription: "+str(redacted_text))
print("\nDuration of the Call: "+str(duration))
print("\nTranscription Time: "+str(time.time()-start))


# COMMAND ----------

def redact_sequence_patterns(text: str) -> str:
    pattern = r'\b\w+(-\w+)+\b'
    number_pattern = re.compile(r'\b\d+\b')
    pattern = r'\b\w+([.*\-_]+\w+)+\b'
    #alphabet_sequence_pattern = re.compile(r'\b[A-Za-z]+(?=[^\w\s])|\b[A-Za-z]+(?=$)')
    redacted_text = re.sub(pattern, '***', text)
    redacted_text = number_pattern.sub("[REDACTED]", redacted_text)
    redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)
    #redacted_text = alphabet_sequence_pattern.sub("[TRUNCATED]", redacted_text)
    return redacted_text

def truncate_bank_account_numbers(text):
    # Regex pattern to find groups of digits (bank account numbers)
    # Adjust the pattern according to the specific format of the bank account numbers you expect
    pattern = r'\b(?:\d{4}[-\s]?){2,}(?=\d{4})'
    
    # Replacement function to keep the last 4 digits and replace others with 'X's
    def repl(match):
        account = match.group(0)
        # Replace all but the last 4 digits with 'X'
        return re.sub(r'\d', 'X', account[:-4]) + account[-4:]
    
    # Replace found bank account numbers with the modified version
    redacted_text = re.sub(pattern, repl, text)
    
    return redacted_text

red = redact_sequence_patterns(redacted_text)
red = truncate_bank_account_numbers(red)
red


# COMMAND ----------

# MAGIC %md
# MAGIC Execution Optimized With Parallel Pool Processing

# COMMAND ----------

num_workers = 4

import os

# Define the list of folder paths
folder_paths = ["/Workspace//Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_2024","/Workspace//Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/jan_2024","/Workspace//Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/production_calls"]

# Initialize an empty list to store the required file paths
files_reqd = []

# Iterate over each folder path
for folder_path in folder_paths:
    # Get the list of all files in the folder
    all_files_in_folder = os.listdir(folder_path)
    
    # Create a dictionary to keep track of file name occurrences
    file_counts = {}
    
    # Iterate over the files in the folder
    for file_name in all_files_in_folder:
        complete_file_path = os.path.join(folder_path, file_name)
        # Check if the file ends with ".wav"
        if file_name.endswith(".wav"):
            # Extract the file name without extension
            file_base_name, file_extension = os.path.splitext(file_name)
            # Check if the file name already exists in the dictionary
            if file_base_name in file_counts:
                # Increment the count and update the file name
                file_counts[file_base_name] += 1
                new_file_name = f"{file_base_name} ({file_counts[file_base_name]}){file_extension}"
            else:
                # Add the file name to the dictionary with count 1
                file_counts[file_base_name] = 1
                new_file_name = file_name
            # Append the complete file path with the updated file name
            files_reqd.append(os.path.join(folder_path, new_file_name))

# Print the list of required files
print(files_reqd)


def redact_sequence_patterns(text: str) -> str:
    pattern = r'\b\w+(-\w+)+\b'
    number_pattern = re.compile(r'\b\d+\b')
    pattern = r'\b\w+([.*\-_]+\w+)+\b'
    redacted_text = re.sub(pattern, '***', text)
    redacted_text = number_pattern.sub("[REDACTED]", redacted_text)
    redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)
    return redacted_text

def truncate_bank_account_numbers(text):

    pattern = r'\b(?:\d{4}[-\s]?){2,}(?=\d{4})'

    def repl(match):
        account = match.group(0)
        return re.sub(r'\d', 'X', account[:-4]) + account[-4:]

    redacted_text = re.sub(pattern, repl, text)
    
    return redacted_text

def transcribe_file(file_path):
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    print("Duration: " + str(duration))
    start_transcription = time.time()    
    tra = ap_distil_med.faster_transcriber(y)
    transcriptions_text = "".join([t['text'] for t in tra])
    start_redaction = time.time()    
    redacted_text = ap_distil_med.double_redact(ap_distil_med.redact(transcriptions_text))
    redacted_text = redact_sequence_patterns(redacted_text)
    redacted_text = truncate_bank_account_numbers(redacted_text)    
    start_summarization = time.time()
    summary = summ.summary_generator(str(tra))
    return {
        'SegmentedTranscription': tra,  
        'Transcription': transcriptions_text,
        'Redacted': redacted_text,
        'Duration': duration,
        'TranscriptionTime': time.time()-start_transcription,
        'RedactionTime': time.time()-start_redaction,
        'SummarizationTime': time.time()-start_summarization,                 
        'Summary': summary
    }

with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
    start = time.time()
    results = list(executor.map(transcribe_file, files_reqd))
    print(results)
    print(f"Transcription, Grammar Correction, and Redaction Time for {str(len(files_reqd))} calls: " + str(time.time() - start))

#with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
#    start = time.time()
#    results = list(executor.map(transcribe_file, files_reqd))
#    print(results)
#    print(f"Transcription, Grammar Correction, and Redaction Time for {len(files_reqd)} calls: {time.time() - start}")

final_df = pd.DataFrame(results)
db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()        

final_df = TopicGenerator(db = db, 
               abfs_client=abfsClient, 
               pytest_flag=False).topic_generator()    

fdf = sentiment_eval(db = db, 
               abfs_client=abfsClient, 
               pytest_flag=False).sentiment_emotion_classifier()

db.write_df_to_azure(abfsClient,
                     input_file=fdf, 
                     azure_path=r'datascience/data/ds/sandbox/shibushaun/silver/final/final_output_combined_distil_med.csv', 
                     format="csv", 
                     verbose=True)


# COMMAND ----------

# MAGIC %md
# MAGIC **Llama2 Inferencing**

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
import concurrent.futures
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

from transformers import pipeline

spacy.cli.download("en_core_web_sm")
nltk.download("punkt")

from huggingface_hub import login
login()

# COMMAND ----------

db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()

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
    topic_output_validated = llama_inference(prompt_template=llama_2_output_structure_confirmation_template.format(t))
    return topic_output_validated#[0]['generated_text'].strip()

def benchmarking(prompt, transcription):
    llama_chat_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    input_prompt = prompt.format(transcription)

    start = time.time()
    output = summary(transcription,
                    prompt)
    exectime = time.time()-start

    print("Length of INPUT PROMPT: "+str(len(llama_chat_tokenizer.tokenize(input_prompt))))
    print("Generated Output: "+output)
    print("Length of INFERENCE: "+str(len(llama_chat_tokenizer.tokenize(output))))

    return {'input_prompt_length': str(len(llama_chat_tokenizer.tokenize(input_prompt))), 
            'output': output, 
            'output_inference_length': str(len(llama_chat_tokenizer.tokenize(output))),
            'execution_time': str(exectime)
            }
    
def redact_sequence_patterns(text: str) -> str:
    pattern = r'\b\w+(-\w+)+\b'
    number_pattern = re.compile(r'\b\d+\b')
    pattern = r'\b\w+([.*\-_]+\w+)+\b'
    redacted_text = re.sub(pattern, '***', text)
    redacted_text = number_pattern.sub("[REDACTED]", redacted_text)
    redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)
    return redacted_text

def truncate_bank_account_numbers(text):
    pattern = r'\b(?:\d{4}[-\s]?){2,}(?=\d{4})'

    def repl(match):
        account = match.group(0)
        return re.sub(r'\d', 'X', account[:-4]) + account[-4:]

    redacted_text = re.sub(pattern, repl, text)
    
    return redacted_text

llama_2_output_structure_confirmation_template = """
[INST] <<SYS>> You are tasked with validating the output format structure for LLAMA 2, an advanced language model. Your role is to confirm that the output from the LLAMA 2 model aligns with the expected format. 
The task involves receiving a topic output and ensuring it is formatted correctly. The expected output should be a Python list containing strings representing topics identified from the conversation content.
Example of the desired output structure: ['topic1', 'topic2', 'topic3', ...].
Your response should consist solely of the validated list of topics. DO NOT ADD ANY ADDITIONAL WORDS OR CHARACTERS.
<<SYS>> {} [/INST]
"""

# COMMAND ----------

with abfsClient.open(r'datascience/data/ds/sandbox/shibushaun/silver/final/final_output_combined_distil_med.csv',"rb") as f:
    test = pd.read_csv(f)

prompt_dict = {}
for task in ["actions_taken", "topic_modelling", "satisfaction_score", "summary"]:
    reloaded_prompt_table = pd.read_csv("/Workspace/Users/sshibu@pplweb.com/RAG_Chatbot/"+task+"_prompts.csv")  
    prompt_dict[task+'_prompt_template'] = reloaded_prompt_table.at[0,'Prompt']

satisfaction_score_prompt_template = prompt_dict['satisfaction_score_prompt_template']
actions_taken_prompt_template = prompt_dict['actions_taken_prompt_template']
topic_inference_prompt_template = prompt_dict['topic_modelling_prompt_template']
summary_prompt_template = prompt_dict['summary_prompt_template']

test['Transcription'] = test['Transcription'].astype(str)
start_time = time.time()
test['llama_inference'] = test['Transcription'].apply(lambda x: llama_call_info_inferencer(x, satisfaction_score_prompt_template, summary_prompt_template, topic_inference_prompt_template, actions_taken_prompt_template) if isinstance(x, str) and len(x.split()) > 50 else {"topics_discussed": "", "actions": "", "summary": "", "satisfaction_dict": ""})
print("Inference completed in:", time.time() - start_time)


# COMMAND ----------

test['llama_actions'] = test['llama_inference'].apply(lambda x: x.get("actions_dict", None))
test['llama_summary'] = test['llama_inference'].apply(lambda x: x.get("summ", None))
test['llama_satisfaction_dict'] = test['llama_inference'].apply(lambda x: x.get("satisfaction_dict", "unidentified"))
test = test[~(test['llama_inference']=={'topics_discussed': '', 'actions': '', 'summary': '', 'satisfaction_dict': ''})]
test['llama_topics_discussed'] = test['llama_inference'].apply(lambda x: x['topics_discussed'][0]['generated_text'])


# COMMAND ----------

test['topic_list'] = test['llama_topics_discussed'].apply(
    lambda x: topic_output_format_validator(x, llama_2_output_structure_confirmation_template)
)
ap = ap_distil_med

test['redacted_summary'] = test['llama_summary'].apply(lambda x: truncate_bank_account_numbers(redact_sequence_patterns(ap.double_redact(ap.redact(x)))) if x is not None else "") 
test['redacted_satisfaction_dict'] = test['llama_satisfaction_dict'].apply(lambda x: truncate_bank_account_numbers(redact_sequence_patterns(ap.double_redact(ap.redact(x)))) if x is not None else "")
test['redacted_actions'] = test['llama_actions'].apply(lambda x: truncate_bank_account_numbers(redact_sequence_patterns(ap.double_redact(ap.redact(x)))) if x is not None else "")

db.write_df_to_azure(
    abfsClient,
    input_file=test,
    azure_path=r'datascience/data/ds/sandbox/shibushaun/silver/final/llama_inference_prod_final_output_combined_distil_med.csv',
    format="csv",
    verbose=True
)


# COMMAND ----------



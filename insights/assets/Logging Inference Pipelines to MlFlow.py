# Databricks notebook source


# COMMAND ----------


!pip install mlflow
!pip install faster-whisper

import faster-whisper
import gc 

# COMMAND ----------



from cs_pa_nlp import (
    AllContextWindowSummaryGenerator,
    AudioProcessor,
    DBUtilConnectionCreator,
)


# COMMAND ----------

!pip install git+https://github.com/m-bain/whisperx.git

# COMMAND ----------

import torch
from torch import device
import pandas as pd

db = DBUtilConnectionCreator(dbutils=dbutils)
abfsClient = db.get_abfs_client()
##READING TRANSCRIPTIONS FOR INFERENCE##
with abfsClient.open(r'datascience/data/ds/sandbox/shibushaun/silver/final/llama_inference_prod_final_output_combined_medium_FEB.csv',"rb") as f:
    test = pd.read_csv(f)
    

# COMMAND ----------

num_rows = 3
calls = test.iloc[:num_rows,:]
df = spark.createDataFrame(calls) 

# COMMAND ----------

# MAGIC %md
# MAGIC Whisper

# COMMAND ----------

import huggingface_hub
import mlflow
import os
import ctranslate2
import librosa
import time
import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from faster_whisper import WhisperModel, utils
import time
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# COMMAND ----------

class Transcribe(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.transcription_model = WhisperModel("distil-medium.en",
                                                device="cpu",
                                                compute_type="float32")
    def predict(self, context, model_input):
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
                model_input,
                beam_size=5,#5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=30)
                #vad_parameters=dict(min_silence_duration_ms=500)
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
    
transcription_model = WhisperModel("distil-medium.en",
                                   device="cpu",
                                   compute_type="float32")

#transcription_model.save("./transcription_pipeline")

model_path = utils.download_model(size_or_id="distil-medium.en", output_dir="./transcription_pipeline")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(artifact_path="transcription_pipeline",
                            python_model=Transcribe(),
                            artifacts={"transcription_pipeline": "./transcription_pipeline"})

model_uri = "runs:/{run}/transcription_pipeline".format(run = run.info.run_id)
model = mlflow.pyfunc.load_model(model_uri)

audio_path = "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080843.wav" 
start = time.time()
y,sr = librosa.load(audio_path)
predictions = model.predict(y)
print(predictions)
print("Transcription Complete: "+str(time.time()-\
    start))

# COMMAND ----------

device = "cpu" 
audio_file = audio_path
batch_size = 8 # reduce if low on GPU mem
compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("distil-medium.en", 
                            device, 
                            compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

start = time.time()

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, 
                          batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], 
                                              device=device)
result = whisperx.align(result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        device, 
                        return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", 
                                             device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, 
                                       result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs

print("Transcription Complete: "+str(time.time()-start))

# COMMAND ----------

def transcribe(audio):
    model_uri = "runs:/{run}/transcription_pipeline".format(run = run.info.run_id)
    model = mlflow.pyfunc.load_model(model_uri)
    y,sr = librosa.load(audio)
    start = time.time()
    predictions = model.predict(y)
    return(predictions)
    #print("Time to process call: "+str(time.time()-start))

file_names = ["/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080843.wav", "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080864.wav", "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080944.wav", "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080960.wav"]
rdd = spark.sparkContext.parallelize(file_names)
print("Parallelization has started")
start = time.time()
result = rdd.map(lambda x: transcribe(x))
results = result.collect()
print(results)     
print("Parallelization Complete: "+str(time.time()-start))

# COMMAND ----------

# Define a UDF wrapping the transcribe function
transcribe_udf = udf(transcribe, StringType())

# List of file names (assuming your list of files)
file_names = [
    "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080843.wav",
    "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080864.wav",
    "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080944.wav",
    "/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/feb_week_of_19_to_23/6080960.wav"
]

# Create a DataFrame with the file names
df = spark.createDataFrame(file_names, StringType()).toDF("file_path")

start = time.time()

results_df = df.withColumn("transcription", transcribe_udf(col("file_path")))

results_df.show()

results = results_df.collect()

print("Parallelization Complete: "+str(time.time()-start))

# COMMAND ----------

results_df.show()

# COMMAND ----------

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import librosa
import mlflow.pyfunc

# Assuming `file_paths` is a list of your audio file paths
ddf = dd.from_pandas(pd.DataFrame(file_paths, columns=['file_path']), npartitions=10)
ddf['transcription'] = ddf['file_path'].map(transcribe, meta=('transcription', str))

with ProgressBar():
    result = ddf.compute()


# COMMAND ----------

# MAGIC %md
# MAGIC Grammar Corrector
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import time

class Gramformer:
    def __init__(self, 
                 models=1, 
                 use_gpu=False, 
                 #db="", 
                 #abfsClient="",
                 correction_model = "",
                 correction_tokenizer = "",                 
                 ):
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
        self.annotator = errant.load('en')
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = device
        self.model_loaded = False

        if models == 1:
            self.correction_model, self.correction_tokenizer = correction_model, correction_tokenizer 
            self.correction_model = self.correction_model.to(device)
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
            input_ids = self.correction_tokenizer.encode(
                input_sentence, return_tensors='pt')
            input_ids = input_ids.to(self.device)

            preds = self.correction_model.generate(
                input_ids,
                do_sample=True,
                max_length=128, 
                num_beams=7,
                early_stopping=True,
                num_return_sequences=max_candidates)

            corrected = set()
            for pred in preds:
                corrected.add(self.correction_tokenizer.decode(
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
                        " " + edit_str_end + "'>" + new_edit_str + "</a>"
                orig_tokens[edit_spos] = st
            elif edit_str_end == "":
                st = "<d type='" + edit_type + "' edit=''>" + edit_str_start + "</d>"
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
    
def grammar_correct(gf_model, 
                      transcription):
    """
    Correct grammar in a given transcription.

    This method takes a transcription as input, 
    splits it into sentences, and attempts to correct the grammar for each
    sentence using a grammar correction tool (gf_model). 
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
                gf_model.correct(
                    influent_sentence, 
                    max_candidates=1))  # Convert set to list
            corrected_sentence = corrected_sentences[0] if corrected_sentences else influent_sentence
            corrected_transcription += "".join(corrected_sentence)
            
            corrected_transcription += " "

        return corrected_transcription
    except Exception as e:
        self.logger.error(f'Error in grammar_corrector: {str(e)}')
        self.write_log_to_azure()
        print(f"Error in grammar_corrector: {str(e)}")
    
class GramformerModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.correction_tokenizer = AutoTokenizer.from_pretrained(context.artifacts["pipeline"])
        self.correction_model = AutoModelForSeq2SeqLM.from_pretrained(context.artifacts["pipeline"])

    def predict(self, context, model_input):
        # Initialize Gramformer with the correction model and tokenizer
        gramformer = Gramformer(models=1, 
                                use_gpu=False, 
                                correction_model=self.correction_model,
                                correction_tokenizer=self.correction_tokenizer)
        # Assume model_input is a single string text for correction
        corrected = gramformer.correct(model_input, max_candidates=1)
        corrected_text = next(iter(corrected), model_input)  # Take the first correction
        return corrected_text
    
correction_model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")   
correction_tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1") 
correction_model.save_pretrained("./pipeline")
correction_tokenizer.save_pretrained("./pipeline")

with mlflow.start_run() as run:        
  mlflow.pyfunc.log_model(artifacts={'pipeline': "./pipeline"}, 
                          artifact_path="gramformer_model", 
                          python_model=GramformerModel())

spark = SparkSession.builder.appName("GrammarCorrection").getOrCreate()
model_uri = "runs:/{}/gramformer_model".format(run.info.run_id)
model = mlflow.pyfunc.load_model(model_uri)
model_uri_broadcast = spark.sparkContext.broadcast(model_uri)

def grammar_correction(transcription):
    # Since model_uri is broadcasted, access it using .value
    model_uri = model_uri_broadcast.value
    model = mlflow.pyfunc.load_model(model_uri)
    # Initialize TextRedactor (or a modified, serializable version of it)
    prediction = model.predict(transcription)
    # Return the redacted text
    return prediction

print("Prediction for one sample: ")
data = "This are a grammatically incorrect sentence"

predictions = model.predict(data)
print(predictions)

print(f"Prediction for {num_rows} samples: ")
start = time.time()
print("Start Time: "+str(time.time()))

grammar_correction_udf = udf(grammar_correction, 
                             StringType())

df_grammar_corrected = df.withColumn("grammar_corrected_transcription", 
                                     grammar_correction_udf(col("Transcription")))

df_grammar_corrected.show()
print(f"Time to evaluate {num_rows} transcriptions: "+str(time.time()-start))

# COMMAND ----------

# MAGIC %md
# MAGIC PII Redaction

# COMMAND ----------

# MAGIC %md
# MAGIC Redaction 1

# COMMAND ----------

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col 
from pyspark.sql.types import StringType
import mlflow.pyfunc
import time

class PIIRoberta(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = pipeline(
            "token-classification",
            model="Jean-Baptiste/roberta-large-ner-english" ,
            aggregation_strategy="simple")
        
    def predict(self, context, model_input):
        predictions = self.model(model_input)
        return predictions
    

class TitlesRecognizer(PatternRecognizer):
    def __init__(self):
        patterns = [r"\bMr\.\b", r"\bMrs\.\b", r"\bMiss\b"]
        super().__init__(supported_entity="TITLE", deny_list=patterns)


class HFTransformersRecognizer(EntityRecognizer):
    def __init__(self,
                 model_uri,
                 supported_entities,
                 supported_language="en"):
        model = mlflow.pyfunc.load_model(model_uri)
        self.pipeline = mlflow.pyfunc.load_model(model_uri)
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language)

    def load(self):
        pass

    def analyze(self, text, entities=None, nlp_artifacts=None):
        results = []
        predictions = self.pipeline.predict(text)
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
    def __init__(self, model_uri):
        self.analyzer = self.initialize_analyzer(model_uri)
        self.anonymizer = AnonymizerEngine()

    def initialize_analyzer(self, model_uri):
        titles_recognizer = TitlesRecognizer()
        transformers_recognizer = HFTransformersRecognizer(model_uri,
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

model_id_or_path = "Jean-Baptiste/roberta-large-ner-english"  
model = pipeline(
            "token-classification",
            model=model_id_or_path,
            aggregation_strategy="simple")

model.save_pretrained("./pii_ner_pipeline")

with mlflow.start_run() as run:        
  mlflow.pyfunc.log_model(artifacts={'pii_ner_pipeline': "./pii_ner_pipeline"}, 
                          artifact_path="jb_roberta_ner", 
                          python_model=PIIRoberta())

spark = SparkSession.builder.appName("PIIRedactionOne").getOrCreate()
model_uri = "runs:/{}/jb_roberta_ner".format(run.info.run_id)
model = mlflow.pyfunc.load_model(model_uri)
model_uri_broadcast = spark.sparkContext.broadcast(model_uri)

text = """ Hi, this is Gina. How can I help you today? Yeah, Gina. I'm Jimmy Marlar. I'm with Interstate Batteries. We have a warehouse at 2001 Interchange Way, Quaker Town, Pennsylvania. Okay. And I've got two accounts already set up, but I was just notified or told by our operations manager that we have a third meter on our building. I need to find out what that is for. And I've got the meter number. Alright. Oh, perfect! Alright, to me first, can you give me your last name again, please? Marler. M-A-R-L-E-R. Thank you. Because two of the meters that I already have in the account, the two meters I have in the account for, one's the fire pump and one's the warehouse, so I don't know unless it's for office. Are you air conditioning or something like, but I have no clue, and the number is 300-466919. And when I put it into the website, it shows 2001 interchange way, but I don't know what it supplies. Now, it's supposed to be an interstate battery's name? Yes, but it's probably under NFI, which is the landlord's name. Alright, let's see. Three account, which means it's large power. I can't, let me see if I can find out. Give me a minute. Did you say it's for large power? Mm-hmm. Okay, that may be the upgraded service we put in. It is, there was a work order. That's what I just got to, so let's see. Give me a second. There's a work order involved that was completed. Let me take a look. The meter number you gave me, this is all. This is the large power, that's all I could say. I have here, it says PPL to install two more sets, and I don't know if this means anything, mate, but I don't know, because I don't do three-phase. It says 750AL and MLP3-PTSWY. So this is the large power one. Okay. All right. So I need to get that put under our name then, don't I? All right. Let me ask you one last question so I know where I'm going. This is in a person's name. Do you know who owned the building? Uh, is it John Arnett? Perfect. Yes. Yes. Yeah, that's who we're dealing with, he's our landlord, he's part of the company that owns the building that we're leasing from, but I just, he emailed this morning saying, Hey, we got a third meter that y'all probably need to get power or over y'alls name. Is there any way we can backdate to January 1st when the other two services went into effect? We're not able to go back. Okay, so I need to do this. Let me tell you, this went into his name, give me a minute, so the connect was completed November 30th, so the first bill he got, give me a minute, was from November 30th to January 4th, he got one bill. Anything, yeah, so, and then, now we're gonna bill, we did bill up to February 2nd, there's a note on here that there's billing in progress, they're billing up to, yeah, so the next bill he gets would probably, I would just say between you and him, to be honest with you. Yeah. I can't go backwards because it's already going to be long. So if I want to put start service day, could I put it as today, would it go through today, or would it be better to put tomorrow as this? Tomorrow. Okay. It's always the next day that we'll do it for. Alright. Okay. We'll do that. Give me a second. Let me just. Full business. find one of your accounts and give me a minute I can do this I don't have to do it online oh no I could do it for you if you want me to it's up to you I mean I Alright, so we have, alright, are you online now? Yeah, I'm, I actually, uh, point of contact, yeah. Yeah, you could, sure, I'll wait, make sure it goes through for ya. Online, honestly, wasn't gonna allow you to do it either. It would always give you the, it'll always give you the next business day on mine, um, but, it'll, you'll see when you get there, but, until I get to the end. Actually, it's asking me for the text ID number. If, if you activate it, you can just mirror what's on the other accounts, correct? Exactly. Yes, I can, uh-huh. I will let you do that. Alright, then let me do it. I'll give you the account numbers. Alright, so it's interstate batteries. Yep, perfect. Go ahead. Okay, the... I believe this is the fire pump. It is six four five seven seven dash six three zero one eight. The fire pump. Okay. And meter number... Let's verify the meter again. Do you still have it in front of you, Jimmy? Uh... let me... let me go back to it now. I like to do that when I have a lot of accounts open in front of me. Just to make sure I'm doing the right one. Yes. Let me see. Where'd it go? Where'd it go? Robert, Robert. No, I'm okay. I'm all right, because I just went back to John's account and took it off of there. We're okay. Okay. Because it's the only one. We're all right. Okay. All right, so we're doing this for tomorrow, the 13th. It's all business use. Yep. and just verify the mailing address for me it's in front of me so you could just verify it for like invoices yes exactly correct okay it's i believe it's my address here in dallas i'm in Texas. Mm-hmm. Does it? Is it? Okay. That's what I have a Texas address. Okay. Exactly. Let me. All right. Let me. Let me get it. I've got so many addresses I can't remember. It is 14221 Dallas Parkway Suite 1000, Dallas, Texas 75254. Perfect. Perfect. All right. So I copied that, seven, two, five, seven, five, two. The zip, wait a minute, zip code is seven, five, two, five, four. Yes, ma'am. All right, I didn't know if I transposed my numbers. All right, just verify what phone number should be on the account for me. Two, one, four, four, four, nine, three, six, six, six. Alright, and then I have your email address. We'll leave this, we'll leave it on this account also, right? Yes, for now. We'll probably end up switching it. We've got an operations manager up in Quaker Town now, so. okay all right so we will start the service at 2001 interchange way in Quaker town effective tomorrow February 13th in the name of interstate batteries right now let me go through a few more things so security deposit is is not required for new service requests, but PPL may require a security deposit in the future if bill payment is received late. Okay. And then, give me one last thing. All right. Now, as a PPL electric utility customer, you're entitled to certain programs and alerts, so I'm just gonna quickly set up the account so that we can efficiently assist you any future needs. Now do you want to add an alternate phone number to the account? Not yet. All right and that's okay and right now how do you prefer to be contacted? Email or not? Email. Okay all right now ask for alerts please know that alerts can be sent at any time of the day or night you will receive the my PPL alerts terms and conditions and its entirety via the US mail at the address on file for the account which is the Texas address and then you may unsubscribe to these alerts to the PPL's website or by contacting PPL at 1-800-342-5775. So do I have your permission to enroll you in my PPL alerts. Yeah. Alrighty. And that'll be through email also. Alright. Now, as part of... Oh, I see you're enrolled in paperless billing on the previous account. We'll continue this service on your new account. Alright. Yeah. Um, we also offer automatic bill pay. Would you be interested in enrolling in automatic bill pay, where we would deduct a payment from a checking or savings account on the due date? No, not right now. Okay. All right, do you want to write down the new account number? Uh, yeah, let me, let me just do that. Okay. The new account number is 9-4-5-8-6-dash-6-7-0-1-8. Oh, okay. And then just so you know, to make it easy for you to access your account online, we did link your PPL account to jimmy.marleratibsa.com. So check your email. I'm the manager account. All right, so you are all set and have I satisfied your concerns today? All right, will you take care you have a good evening. Thank you. Thank you. Bye bye"""

print("Prediction for one sample using the logged model")
start = time.time()
print(str(start))
redacted_text = TextRedactor(model_uri=model_uri).redact_text(input_text=text)
print(redacted_text)
print("Time for Inference: "+str(time.time()-start))

def pii_redaction_one(text):
    model_uri = model_uri_broadcast.value
    model = mlflow.pyfunc.load_model(model_uri) 
    redactor = TextRedactor(model_uri) 
    return redactor.redact_text(text)

print(f"Prediction for {num_rows} samples: ")
start = time.time()
print("Start Time: "+str(start))

pii_redaction_udf = udf(pii_redaction_one, 
                        StringType())

df_redacted = df.withColumn("redacted_transcription",
                            pii_redaction_udf(col("Transcription")))

df_redacted.show()

print(f"Time to evaluate {num_rows} transcriptions: "+str(time.time()-start))


# COMMAND ----------

# MAGIC %md
# MAGIC Redaction 2

# COMMAND ----------

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col 
from pyspark.sql.types import StringType
import mlflow.pyfunc
import time

class PIIStanford(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.drm_path = "StanfordAIMI/stanford-deidentifier-base"
        self.drt = AutoTokenizer.from_pretrained(self.drm_path)
        self.drm = AutoModelForTokenClassification.from_pretrained(self.drm_path)

    def predict(self, context, model_input):
        redact_pii_pipeline = pipeline("ner",
                                        model=self.drm,
                                        tokenizer=self.drt,
                                        aggregation_strategy='average')
        pii_words = [item['word'] for item in redact_pii_pipeline(model_input)]
        modified_string = model_input
        for pii_word in pii_words:
            modified_string = re.sub(r'\b' + re.escape(pii_word) + r'\b', '*' * len(pii_word), modified_string)
        return modified_string
    
tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/stanford-deidentifier-base")
model = AutoModelForTokenClassification.from_pretrained("StanfordAIMI/stanford-deidentifier-base")
model.save_pretrained("./pipeline")
tokenizer.save_pretrained("./pipeline")

with mlflow.start_run() as run:        
  mlflow.pyfunc.log_model(artifacts={'pipeline': "./pipeline"}, 
                          artifact_path="stanford_model", 
                          python_model=PIIStanford())

spark = SparkSession.builder.appName("PIIRedaction2").getOrCreate()
model_uri = "runs:/{}/stanford_model".format(run.info.run_id)
model = mlflow.pyfunc.load_model(model_uri)
model_uri_broadcast = spark.sparkContext.broadcast(model_uri)

def pii_redaction_two(transcription):
    model_uri = model_uri_broadcast.value
    model = mlflow.pyfunc.load_model(model_uri)
    prediction = model.predict(transcription)
    return prediction

print("Prediction for one sample: ")
data = "Shaun Shibu lives in Jersey City and his phone number is 1112223322."

predictions = model.predict(data)
print(predictions)

print(f"Prediction for {num_rows} samples: ")
start = time.time()
print("Start Time: "+str(time.time()))

pii_redaction_two_udf = udf(pii_redaction_two, 
                             StringType())

df_redacted = df.withColumn("redacted_transcription", 
                            pii_redaction_two_udf(col("Transcription")))

df_redacted.show()
print(f"Time to evaluate {num_rows} transcriptions: "+str(time.time()-start))

# COMMAND ----------

# MAGIC %md
# MAGIC Redaction 3

# COMMAND ----------

def redact_sequence_patterns(text: str) -> str:
    pattern = r'\b\w+(-\w+)+\b'
    number_pattern = re.compile(r'\b\d+\b')
    pattern = r'\b\w+([.*\-_]+\w+)+\b'
    redacted_text = re.sub(pattern, '***', text)
    redacted_text = number_pattern.sub("[REDACTED]", redacted_text)
    redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)
    pattern = r'\b(?:\d{4}[-\s]?){2,}(?=\d{4})'
    def repl(match):
        account = match.group(0)
        # Replace all but the last 4 digits with 'X'
        return re.sub(r'\d', 'X', account[:-4]) + account[-4:]
    redacted_text = re.sub(pattern, repl, redacted_text)
    return redacted_text

print("Prediction for one sample using the logged model")
start = time.time()
print(str(start))
final = redact_sequence_patterns(predictions)
print(final)
print("Time for Inference: "+str(time.time()-start))

print(f"Prediction for {num_rows} samples: ")
start = time.time()
print("Start Time: "+str(start))

pii_redaction_udf = udf(redact_sequence_patterns, 
                        StringType())

df_redacted = df_redacted.withColumn("redacted_transcription",
                                     pii_redaction_udf(col("redacted_transcription")))

df_redacted.show()

print(f"Time to evaluate {num_rows} transcriptions: "+str(time.time()-start))


# COMMAND ----------

# MAGIC %md
# MAGIC Summarization

# COMMAND ----------

data = """
[{'start': 1.71, 'end': 5.41, 'text': ' Hi, this is Gina. How can I help you today?', 'no_speech_probability': 0.06262744218111038, 'sentiment': 'neutral'}, {'start': 6.46, 'end': 11.96, 'text': " Yeah, Gina. I'm Jimmy Marlar. I'm with Interstate Batteries.", 'no_speech_probability': 0.06262744218111038, 'sentiment': 'neutral'}, {'start': 11.96, 'end': 20.82, 'text': ' We have a warehouse at 2001 Interchange Way, Quaker Town, Pennsylvania.', 'no_speech_probability': 0.06262744218111038, 'sentiment': 'neutral'}, {'start': 21.42, 'end': 22.22, 'text': ' Okay.', 'no_speech_probability': 0.06262744218111038, 'sentiment': 'neutral'}, {'start': 22.22, 'end': 38.6, 'text': " And I've got two accounts already set up, but I was just notified or told by our operations manager that we have a third meter on our building.", 'no_speech_probability': 0.05263296514749527, 'sentiment': 'neutral'}, {'start': 38.6, 'end': 46.7, 'text': " I need to find out what that is for. And I've got the meter number.", 'no_speech_probability': 0.05263296514749527, 'sentiment': 'neutral'}, {'start': 46.7, 'end': 54.18, 'text': ' Alright. Oh, perfect! Alright, to me first, can you give me your last name again, please?', 'no_speech_probability': 0.33603402972221375, 'sentiment': 'positive'}, {'start': 54.18, 'end': 58.17, 'text': ' Marler. M-A-R-L-E-R.', 'no_speech_probability': 0.33603402972221375, 'sentiment': 'neutral'}, {'start': 58.17, 'end': 60.14, 'text': ' Thank you.', 'no_speech_probability': 0.33603402972221375, 'sentiment': 'positive'}, {'start': 60.14, 'end': 80.2, 'text': " Because two of the meters that I already have in the account, the two meters I have in the account for, one's the fire pump and one's the warehouse, so I don't know unless it's for office.", 'no_speech_probability': 0.33603402972221375, 'sentiment': 'neutral'}, {'start': 80.2, 'end': 102.0, 'text': ' Are you air conditioning or something like, but I have no clue, and the number is 300-466919.', 'no_speech_probability': 0.11389932781457901, 'sentiment': 'neutral'}, {'start': 102.0, 'end': 122.89, 'text': " And when I put it into the website, it shows 2001 interchange way, but I don't know what it supplies.", 'no_speech_probability': 0.11389932781457901, 'sentiment': 'neutral'}, {'start': 122.89, 'end': 129.1, 'text': " Now, it's supposed to be an interstate battery's name?", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 129.1, 'end': 135.84, 'text': " Yes, but it's probably under NFI, which is the landlord's name.", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 135.84, 'end': 153.82, 'text': " Alright, let's see.", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 153.82, 'end': 158.71, 'text': " Three account, which means it's large power.", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 158.71, 'end': 161.71, 'text': " I can't, let me see if I can find out.", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 161.71, 'end': 167.58, 'text': ' Give me a minute.', 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 167.58, 'end': 171.96, 'text': " Did you say it's for large power?", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 171.96, 'end': 174.14, 'text': ' Mm-hmm.', 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 174.14, 'end': 179.14, 'text': ' Okay, that may be the upgraded service we put in.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 182.25, 'end': 184.25, 'text': ' It is, there was a work order.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 184.25, 'end': 188.53, 'text': " That's what I just got to, so let's see.", 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 188.53, 'end': 189.53, 'text': ' Give me a second.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 189.53, 'end': 195.03, 'text': " There's a work order involved that was completed.", 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 195.03, 'end': 213.49, 'text': ' Let me take a look.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 213.49, 'end': 238.9, 'text': ' The meter number you gave me, this is all.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 238.9, 'end': 242.06, 'text': " This is the large power, that's all I could say.", 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 242.06, 'end': 247.06, 'text': ' I have here, it says PPL to install two more sets,', 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 249.14, 'end': 251.18, 'text': " and I don't know if this means anything,", 'no_speech_probability': 0.10594843327999115, 'sentiment': 'negative'}, {'start': 251.18, 'end': 252.94, 'text': " mate, but I don't know,", 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 252.94, 'end': 254.7, 'text': " because I don't do three-phase.", 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 254.7, 'end': 259.74, 'text': ' It says 750AL and MLP3-PTSWY.', 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 264.94, 'end': 270.3, 'text': ' So this is the large power one.', 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 270.3, 'end': 279.1, 'text': " Okay. All right. So I need to get that put under our name then, don't I?", 'no_speech_probability': 0.2681259512901306, 'sentiment': 'neutral'}, {'start': 279.1, 'end': 284.1, 'text': " All right. Let me ask you one last question so I know where I'm going.", 'no_speech_probability': 0.2681259512901306, 'sentiment': 'neutral'}, {'start': 284.1, 'end': 291.61, 'text': " This is in a person's name. Do you know who owned the building?", 'no_speech_probability': 0.2681259512901306, 'sentiment': 'neutral'}, {'start': 291.61, 'end': 297.76, 'text': ' Uh, is it John Arnett?', 'no_speech_probability': 0.2681259512901306, 'sentiment': 'neutral'}, {'start': 297.76, 'end': 301.76, 'text': ' Perfect. Yes. Yes.', 'no_speech_probability': 0.2681259512901306, 'sentiment': 'positive'}, {'start': 301.76, 'end': 323.75, 'text': " Yeah, that's who we're dealing with, he's our landlord, he's part of the company that owns the building that we're leasing from, but I just, he emailed this morning saying,", 'no_speech_probability': 0.43267571926116943, 'sentiment': 'neutral'}, {'start': 323.75, 'end': 329.78, 'text': " Hey, we got a third meter that y'all probably need to get power", 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 329.78, 'end': 334.14, 'text': " or over y'alls name.", 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 334.14, 'end': 343.1, 'text': ' Is there any way we can backdate to January 1st', 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 343.1, 'end': 348.19, 'text': ' when the other two services went into effect?', 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 348.19, 'end': 352.5, 'text': " We're not able to go back.", 'no_speech_probability': 0.10858119279146194, 'sentiment': 'negative'}, {'start': 352.5, 'end': 354.98, 'text': ' Okay, so I need to do this.', 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 354.98, 'end': 397.66, 'text': ' Let me tell you, this went into his name, give me a minute, so the connect was completed November 30th, so the first bill he got, give me a minute, was from November 30th to January 4th, he got one bill.', 'no_speech_probability': 0.27920442819595337, 'sentiment': 'neutral'}, {'start': 397.66, 'end': 409.92, 'text': " Anything, yeah, so, and then, now we're gonna bill, we did bill up to February 2nd, there's", 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 409.92, 'end': 421.6, 'text': " a note on here that there's billing in progress, they're billing up to, yeah, so the next", 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 421.6, 'end': 428.72, 'text': ' bill he gets would probably, I would just say between you and him, to be honest with', 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 428.72, 'end': 430.52, 'text': ' you.', 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 430.52, 'end': 431.52, 'text': ' Yeah.', 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 431.52, 'end': 435.52, 'text': " I can't go backwards because it's already going to be long.", 'no_speech_probability': 0.31251534819602966, 'sentiment': 'negative'}, {'start': 435.52, 'end': 443.56, 'text': ' So if I want to put start service day, could I put it as today, would it go through today,', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 443.56, 'end': 446.56, 'text': ' or would it be better to put tomorrow as this?', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 446.56, 'end': 448.56, 'text': ' Tomorrow.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 448.56, 'end': 449.59, 'text': ' Okay.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 449.59, 'end': 453.59, 'text': " It's always the next day that we'll do it for.", 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 453.59, 'end': 454.59, 'text': ' Alright.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 454.59, 'end': 455.59, 'text': ' Okay.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 455.59, 'end': 456.59, 'text': " We'll do that.", 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 456.59, 'end': 457.59, 'text': ' Give me a second.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 457.59, 'end': 458.59, 'text': ' Let me just.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 458.59, 'end': 459.59, 'text': ' Full business.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 459.59, 'end': 473.42, 'text': " find one of your accounts and give me a minute I can do this I don't have to do", 'no_speech_probability': 0.058932796120643616, 'sentiment': 'neutral'}, {'start': 473.42, 'end': 483.07, 'text': " it online oh no I could do it for you if you want me to it's up to you I mean I", 'no_speech_probability': 0.058932796120643616, 'sentiment': 'neutral'}, {'start': 483.07, 'end': 490.89, 'text': ' Alright, so we have, alright, are you online now?', 'no_speech_probability': 0.5168992280960083, 'sentiment': 'neutral'}, {'start': 490.89, 'end': 499.49, 'text': " Yeah, I'm, I actually, uh, point of contact, yeah.", 'no_speech_probability': 0.5168992280960083, 'sentiment': 'positive'}, {'start': 499.49, 'end': 508.5, 'text': " Yeah, you could, sure, I'll wait, make sure it goes through for ya.", 'no_speech_probability': 0.5168992280960083, 'sentiment': 'positive'}, {'start': 508.5, 'end': 514.5, 'text': " Online, honestly, wasn't gonna allow you to do it either.", 'no_speech_probability': 0.5168992280960083, 'sentiment': 'negative'}, {'start': 514.5, 'end': 533.56, 'text': " It would always give you the, it'll always give you the next business day on mine, um, but, it'll, you'll see when you get there, but, until I get to the end.", 'no_speech_probability': 0.20985403656959534, 'sentiment': 'neutral'}, {'start': 533.56, 'end': 543.56, 'text': " Actually, it's asking me for the text ID number. If, if you activate it, you can just mirror what's on the other accounts, correct?", 'no_speech_probability': 0.20985403656959534, 'sentiment': 'neutral'}, {'start': 543.56, 'end': 547.56, 'text': ' Exactly. Yes, I can, uh-huh.', 'no_speech_probability': 0.20985403656959534, 'sentiment': 'neutral'}, {'start': 547.56, 'end': 551.01, 'text': ' I will let you do that. Alright, then let me do it.', 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 551.01, 'end': 554.05, 'text': " I'll give you the account numbers. Alright, so it's", 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 554.05, 'end': 559.76, 'text': ' interstate batteries. Yep, perfect. Go ahead.', 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 559.76, 'end': 569.83, 'text': ' Okay, the... I believe this is the fire pump. It is', 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 569.83, 'end': 585.62, 'text': ' six four five seven seven dash six three zero one eight.', 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 585.62, 'end': 588.09, 'text': ' The fire pump. Okay.', 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 588.09, 'end': 620.49, 'text': ' And meter number...', 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 620.49, 'end': 624.55, 'text': " Let's verify the meter again. Do you still have it in front of you, Jimmy?", 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 624.55, 'end': 632.39, 'text': ' Uh... let me... let me go back to it now.', 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 632.39, 'end': 644.8, 'text': ' I like to do that when I have a lot of accounts open in front of me.', 'no_speech_probability': 0.3525732457637787, 'sentiment': 'positive'}, {'start': 644.8, 'end': 647.3, 'text': " Just to make sure I'm doing the right one.", 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 647.3, 'end': 654.3, 'text': " Yes. Let me see. Where'd it go? Where'd it go? Robert, Robert.", 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 654.3, 'end': 686.53, 'text': " No, I'm okay. I'm all right, because I just went back to John's account and took it off of there. We're okay.", 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 686.53, 'end': 687.03, 'text': ' Okay.', 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 687.03, 'end': 691.2, 'text': " Because it's the only one. We're all right.", 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 691.2, 'end': 692.2, 'text': ' Okay.', 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 692.2, 'end': 701.96, 'text': " All right, so we're doing this for tomorrow, the 13th. It's all business use.", 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 701.96, 'end': 703.44, 'text': ' Yep.', 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 703.44, 'end': 713.54, 'text': " and just verify the mailing address for me it's in front of me so you could just verify it", 'no_speech_probability': 0.12320521473884583, 'sentiment': 'neutral'}, {'start': 717.25, 'end': 729.22, 'text': " for like invoices yes exactly correct okay it's i believe it's my address here in dallas i'm in", 'no_speech_probability': 0.12320521473884583, 'sentiment': 'neutral'}, {'start': 729.22, 'end': 738.0, 'text': " Texas. Mm-hmm. Does it? Is it? Okay. That's what I have a Texas address. Okay. Exactly.", 'no_speech_probability': 0.6127711534500122, 'sentiment': 'neutral'}, {'start': 738.0, 'end': 748.8, 'text': " Let me. All right. Let me. Let me get it. I've got so many addresses I can't remember.", 'no_speech_probability': 0.6127711534500122, 'sentiment': 'neutral'}, {'start': 750.0, 'end': 765.5, 'text': ' It is 14221 Dallas Parkway Suite 1000, Dallas, Texas 75254. Perfect. Perfect. All right.', 'no_speech_probability': 0.6127711534500122, 'sentiment': 'neutral'}, {'start': 765.74, 'end': 770.74, 'text': ' So I copied that, seven, two, five, seven, five, two.', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 771.74, 'end': 776.74, 'text': ' The zip, wait a minute, zip code is seven, five, two, five, four.', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 778.56, 'end': 781.03, 'text': " Yes, ma'am.", 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 781.03, 'end': 784.87, 'text': " All right, I didn't know if I transposed my numbers.", 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 784.87, 'end': 789.59, 'text': ' All right, just verify what phone number', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 789.59, 'end': 792.55, 'text': ' should be on the account for me.', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 792.55, 'end': 796.55, 'text': ' Two, one, four, four, four, nine, three, six, six, six.', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 798.46, 'end': 805.82, 'text': " Alright, and then I have your email address. We'll leave this, we'll leave it on this account also, right?", 'no_speech_probability': 0.06859400868415833, 'sentiment': 'neutral'}, {'start': 806.3, 'end': 816.77, 'text': " Yes, for now. We'll probably end up switching it. We've got an operations manager up in Quaker Town now, so.", 'no_speech_probability': 0.06859400868415833, 'sentiment': 'neutral'}, {'start': 816.77, 'end': 829.14, 'text': ' okay all right so we will start the service at 2001 interchange way in', 'no_speech_probability': 0.19277390837669373, 'sentiment': 'neutral'}, {'start': 829.14, 'end': 838.44, 'text': ' Quaker town effective tomorrow February 13th in the name of interstate', 'no_speech_probability': 0.19277390837669373, 'sentiment': 'neutral'}, {'start': 838.44, 'end': 843.9, 'text': ' batteries right now let me go through a few more things so security deposit is', 'no_speech_probability': 0.19277390837669373, 'sentiment': 'neutral'}, {'start': 843.9, 'end': 847.38, 'text': ' is not required for new service requests,', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 847.38, 'end': 852.26, 'text': ' but PPL may require a security deposit in the future', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 852.26, 'end': 856.38, 'text': ' if bill payment is received late.', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'negative'}, {'start': 856.38, 'end': 857.22, 'text': ' Okay.', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 857.22, 'end': 861.83, 'text': ' And then, give me one last thing.', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 861.83, 'end': 863.02, 'text': ' All right.', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 863.02, 'end': 867.14, 'text': ' Now, as a PPL electric utility customer,', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 867.14, 'end': 870.3, 'text': " you're entitled to certain programs and alerts,", 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 870.3, 'end': 872.62, 'text': " so I'm just gonna quickly set up the account", 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 872.62, 'end': 874.74, 'text': ' so that we can efficiently assist you', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 874.74, 'end': 881.04, 'text': ' any future needs. Now do you want to add an alternate phone number to the', 'no_speech_probability': 0.3322262167930603, 'sentiment': 'neutral'}, {'start': 881.04, 'end': 892.12, 'text': " account? Not yet. All right and that's okay and right now how do you prefer to", 'no_speech_probability': 0.3322262167930603, 'sentiment': 'neutral'}, {'start': 892.12, 'end': 903.36, 'text': ' be contacted? Email or not? Email. Okay all right now ask for alerts please know', 'no_speech_probability': 0.3322262167930603, 'sentiment': 'neutral'}, {'start': 903.36, 'end': 908.66, 'text': ' that alerts can be sent at any time of the day or night you will receive the', 'no_speech_probability': 0.3322262167930603, 'sentiment': 'neutral'}, {'start': 908.66, 'end': 916.26, 'text': ' my PPL alerts terms and conditions and its entirety via the US mail at the address on', 'no_speech_probability': 0.21256396174430847, 'sentiment': 'neutral'}, {'start': 916.26, 'end': 922.98, 'text': ' file for the account which is the Texas address and then you may unsubscribe to these alerts', 'no_speech_probability': 0.21256396174430847, 'sentiment': 'neutral'}, {'start': 922.98, 'end': 933.74, 'text': " to the PPL's website or by contacting PPL at 1-800-342-5775. So do I have your permission", 'no_speech_probability': 0.21256396174430847, 'sentiment': 'neutral'}, {'start': 933.74, 'end': 937.84, 'text': ' to enroll you in my PPL alerts.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 937.84, 'end': 938.84, 'text': ' Yeah.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 938.84, 'end': 939.84, 'text': ' Alrighty.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'positive'}, {'start': 939.84, 'end': 942.84, 'text': " And that'll be through email also.", 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 942.84, 'end': 944.96, 'text': ' Alright.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 944.96, 'end': 946.96, 'text': ' Now, as part of...', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 946.96, 'end': 951.96, 'text': " Oh, I see you're enrolled in paperless billing on the previous account.", 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 951.96, 'end': 955.96, 'text': " We'll continue this service on your new account.", 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 955.96, 'end': 956.96, 'text': ' Alright.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 956.96, 'end': 957.96, 'text': ' Yeah.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 957.96, 'end': 960.96, 'text': ' Um, we also offer automatic bill pay.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 960.96, 'end': 965.82, 'text': ' Would you be interested in enrolling in automatic bill pay,', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 965.82, 'end': 967.82, 'text': ' where we would deduct a payment', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 967.82, 'end': 970.86, 'text': ' from a checking or savings account on the due date?', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 970.86, 'end': 974.28, 'text': ' No, not right now.', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 974.28, 'end': 976.13, 'text': ' Okay.', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 976.13, 'end': 977.83, 'text': ' All right, do you want to write down', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 977.83, 'end': 980.6, 'text': ' the new account number?', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 980.6, 'end': 997.38, 'text': ' Uh, yeah, let me,', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 997.38, 'end': 1013.88, 'text': ' let me just do that.', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 1013.88, 'end': 1018.22, 'text': ' Okay.', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 1018.22, 'end': 1032.88, 'text': ' The new account number is 9-4-5-8-6-dash-6-7-0-1-8.', 'no_speech_probability': 0.08544127643108368, 'sentiment': 'neutral'}, {'start': 1032.88, 'end': 1036.16, 'text': ' Oh, okay.', 'no_speech_probability': 0.08544127643108368, 'sentiment': 'neutral'}, {'start': 1036.16, 'end': 1051.05, 'text': ' And then just so you know, to make it easy for you to access your account online, we did link your PPL account to jimmy.marleratibsa.com.', 'no_speech_probability': 0.08544127643108368, 'sentiment': 'neutral'}, {'start': 1051.05, 'end': 1052.05, 'text': ' So check your email.', 'no_speech_probability': 0.08544127643108368, 'sentiment': 'neutral'}, {'start': 1052.05, 'end': 1059.78, 'text': " I'm the manager account. All right, so you are all set and have I satisfied your concerns today?", 'no_speech_probability': 0.045305266976356506, 'sentiment': 'neutral'}, {'start': 1061.78, 'end': 1071.42, 'text': ' All right, will you take care you have a good evening. Thank you. Thank you. Bye bye', 'no_speech_probability': 0.045305266976356506, 'sentiment': 'positive'}]"""


# COMMAND ----------

import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col 
from pyspark.sql.types import StringType
import mlflow.pyfunc
import time
import ast

class Summarization(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the tokenizer and model for Summarization
        self.summ_path = "philschmid/bart-large-cnn-samsum"
        self.st = AutoTokenizer.from_pretrained(self.summ_path)

    def preprocess(self, transcriptions):
        merged_transcription = ""
        current_speaker = 1

        for i, transcript in enumerate(ast.literal_eval(transcriptions)):
            text = transcript['text'].strip()
            merged_transcription += f"Speaker {current_speaker}: \"{text}\"\n"

            current_speaker = 1 if current_speaker == 2 else 2

        return (merged_transcription)

    def predict(self, context, input_text):
        input_text = self.preprocess(input_text)
        sp = pipeline("summarization", model=self.summ_path)
        try:
            if len(self.st.tokenize(input_text)) > 1000:
                chunk_length = 1000
                chunks = []
                current_chunk = ""

                for line in sent_tokenize(input_text):
                    encoded_line = self.st.encode(line,
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
                    summary = sp(chunk)
                    final_summary += summary[0]['summary_text'] + ' '

                return (final_summary)

            elif len(self.st.tokenize(input_text)) == 0:
                return ("Summary not generated for this transcription")

            else:
                final_summary = sp(input_text)[0]['summary_text']
                return (final_summary)

        except Exception as e:
            print(f"Error in summary_generator: {str(e)}")
        return (summary)
    
tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
#pipeline = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
tokenizer.save_pretrained("./pipeline")
#pipeline.save_pretrained("./pipeline")

with mlflow.start_run() as run:        
  mlflow.pyfunc.log_model(artifacts={'pipeline': "./pipeline"}, 
                          artifact_path="summarization_model", 
                          python_model=Summarization())

spark = SparkSession.builder.appName("Summarization").getOrCreate()
model_uri = "runs:/{}/summarization_model".format(run.info.run_id)
model = mlflow.pyfunc.load_model(model_uri)
model_uri_broadcast = spark.sparkContext.broadcast(model_uri)

def summarization(transcription):
    model_uri = model_uri_broadcast.value
    model = mlflow.pyfunc.load_model(model_uri)
    prediction = model.predict(transcription)
    return prediction

print("Prediction for one sample: ")

predictions = model.predict(data)
print(predictions)

print(f"Prediction for {num_rows} samples: ")
start = time.time()
print("Start Time: "+str(time.time()))

summarization_udf = udf(summarization, 
                        StringType())

df_summary = df.withColumn("summary", 
                            summarization(col("SegmentedTranscription")))

df_summary.show()
print(f"Time to evaluate {num_rows} transcriptions: "+str(time.time()-start))


# COMMAND ----------

# MAGIC %md
# MAGIC Call Classification

# COMMAND ----------

import json

with open(r"/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/topics/topics.json", "r") as f:
    json_data = f.read()

    data = json.loads(json_data)

data

# COMMAND ----------

import json

with open(r"/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/topics/topics_5.json", "r") as f:
    json_data_5 = f.read()

    data_5 = json.loads(json_data_5)

data_5

# COMMAND ----------

list(set(data.keys()) - set(data_5.keys()))

# COMMAND ----------



# COMMAND ----------

import mlflow.pyfunc
from transformers import pipeline
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import json


class CallClassification(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the tokenizer and model for Summarization
        self.path = "facebook/bart-large-mnli"
        self.classifier = pipeline("zero-shot-classification", model=self.path)
        self.model = SentenceTransformer('llmrails/ember-v1')
        with open(r"/Workspace/Users/sshibu@pplweb.com/CS_PA_NLP/mp3_files/topics/topics_5.json", "r") as f:
            json_data = f.read()

            data = json.loads(json_data)
        self.ci = data
        self.topic_embeddings = self.model.encode(
            (list(self.ci.values())))

    def probability_assignment(self, summary, topic_list):
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

    def apply_probability_assignment(self, topic_list, summary):
        print("row wise probability_assignment started")
        try:
            if len(topic_list) == 0:
                print("Unidentified")
                return "UNIDENTIFIED"
            else:
                probabilities = self.probability_assignment(
                    summary, topic_list)
                return probabilities
        except Exception as e:
            self.logger.error(f'Error in\
                               apply_probability_assignment: {str(e)}')
            self.write_log_to_azure()
            print(f"Error in apply_probability_assignment: {str(e)}")

    def parse_topic_with_probabilities(self, x):
        try:
            if type(x) is dict:
                return x
        except (IndexError, ValueError, SyntaxError):
            pass
        return {'Unidentified': 1}

    def get_primary_topic(self, x):
        try:
            return x[list(x.keys())[1]][0]
        except (IndexError, TypeError):
            return 'Unidentified'

    def get_secondary_topic(self, x):
        try:
            if len(list(x.keys())) > 1:
                return x[list(x.keys())[1]][1]
            else:
                return 'None'
        except (IndexError, TypeError):
            return 'None'
        
    def predict(self, context, summary):
        try:
            index = 0
            threshold = 0.4
            top_2_topics_per_cluster = pd.DataFrame(
                columns=[
                    'Sentence',
                    'Topic',
                    'Position',
                    'cos_sim',
                    'Chunking Strategy'])
            print("chunks")
            chunks = list(summary.split('.'))
            chunks = [sentence for sentence in summary.split(
                '.') if len(sentence.split()) >= 5]
            print("sentence_embeddings")
            sentence_embeddings = self.model.encode(chunks)
            print("Dot Product Computation")
            for i, sentence_embedding in enumerate(sentence_embeddings):
                for topic_num, topic_embedding in enumerate(self.topic_embeddings):
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

            position_wise = top_2_topics_per_cluster.sort_values(by=[
                'Position'], ascending=True)
            if len(position_wise) >= 10:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'].iloc[0:10])
            elif len(position_wise) > 0:
                top_topics = list(position_wise.sort_values(by=[
                    'cos_sim'], ascending=False)['Topic'])
            else:
                top_topics = []

        except Exception as e:
            print(f"Error in topic_modeller: {str(e)}")
            return []

        topic_dict = self.parse_topic_with_probabilities(self.apply_probability_assignment(topic_list = top_topics, summary = summary))
        primary = self.get_primary_topic(x = topic_dict)
        secondary = self.get_secondary_topic(x =topic_dict)

        return [primary, secondary]


# COMMAND ----------

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
model = SentenceTransformer('llmrails/ember-v1')

classifier.save_pretrained("./classification_pipeline")
model.save("./classification_pipeline")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(artifact_path="classification_pipeline",
                            python_model=CallClassification(),
                            artifacts={"classification_pipeline": "./classification_pipeline"})

model_uri = "runs:/{}/classification_pipeline".format(run.info.run_id)
model = mlflow.pyfunc.load_model(model_uri)



# COMMAND ----------

summary = """Speaker 1's residence is 16 Wall Street in Harrisburg, Pennsylvania. Speaker 1 had a power outage due to some downed branches on a wire. Speaker 2's heater and light are not working and his landlord wants to know if there could be a meter reset due to the subsequent repair and damage that had occurred outside. PPL will send the issue to the service department."""

predictions = model.predict(summary)

predictions

# COMMAND ----------

# MAGIC %md
# MAGIC Sentiment Analysis

# COMMAND ----------

import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import re
from nltk.tokenize import sent_tokenize

class Sentiment(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the tokenizer and model for Summarization
        self.sent_path = "cardiffnlp/twitter-roberta-base-sentiment"
        self.pipe = pipeline("text-classification", model=self.sent_path) 

    def predict(self, context, input_text):
        sentiment = segment_sentiment_computer(ast.literal_eval(x)) if len(x) > 0 else "No Sentiment Identified"
        return sentiment
    
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

sent_path = "cardiffnlp/twitter-roberta-base-sentiment"
pipe = pipeline("text-classification", model=self.sent_path)

pipe.save_pretrained("./sentiment_pipeline")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(artifact_path="sentiment_pipeline",
                            python_model=Sentiment(),
                            artifacts={"sentiment_pipeline": "./sentiment_pipeline"})


# COMMAND ----------

import ast

def preprocess(transcriptions):
    merged_transcription = ""
    current_speaker = 1

    for i, transcript in enumerate(ast.literal_eval(transcriptions)):
        text = transcript['text'].strip()
        merged_transcription += f"Speaker {current_speaker}: \"{text}\"\n"

        current_speaker = 1 if current_speaker == 2 else 2

    return (merged_transcription)

model_uri = "runs:/{}/sentiment_pipeline".format(run.info.run_id)
#model = mlflow.pyfunc.load_model(model_uri)
text = """[{'start': 1.71, 'end': 5.41, 'text': ' Hi, this is Gina. How can I help you today?', 'no_speech_probability': 0.06262744218111038, 'sentiment': 'neutral'}, {'start': 6.46, 'end': 11.96, 'text': " Yeah, Gina. I'm Jimmy Marlar. I'm with Interstate Batteries.", 'no_speech_probability': 0.06262744218111038, 'sentiment': 'neutral'}, {'start': 11.96, 'end': 20.82, 'text': ' We have a warehouse at 2001 Interchange Way, Quaker Town, Pennsylvania.', 'no_speech_probability': 0.06262744218111038, 'sentiment': 'neutral'}, {'start': 21.42, 'end': 22.22, 'text': ' Okay.', 'no_speech_probability': 0.06262744218111038, 'sentiment': 'neutral'}, {'start': 22.22, 'end': 38.6, 'text': " And I've got two accounts already set up, but I was just notified or told by our operations manager that we have a third meter on our building.", 'no_speech_probability': 0.05263296514749527, 'sentiment': 'neutral'}, {'start': 38.6, 'end': 46.7, 'text': " I need to find out what that is for. And I've got the meter number.", 'no_speech_probability': 0.05263296514749527, 'sentiment': 'neutral'}, {'start': 46.7, 'end': 54.18, 'text': ' Alright. Oh, perfect! Alright, to me first, can you give me your last name again, please?', 'no_speech_probability': 0.33603402972221375, 'sentiment': 'positive'}, {'start': 54.18, 'end': 58.17, 'text': ' Marler. M-A-R-L-E-R.', 'no_speech_probability': 0.33603402972221375, 'sentiment': 'neutral'}, {'start': 58.17, 'end': 60.14, 'text': ' Thank you.', 'no_speech_probability': 0.33603402972221375, 'sentiment': 'positive'}, {'start': 60.14, 'end': 80.2, 'text': " Because two of the meters that I already have in the account, the two meters I have in the account for, one's the fire pump and one's the warehouse, so I don't know unless it's for office.", 'no_speech_probability': 0.33603402972221375, 'sentiment': 'neutral'}, {'start': 80.2, 'end': 102.0, 'text': ' Are you air conditioning or something like, but I have no clue, and the number is 300-466919.', 'no_speech_probability': 0.11389932781457901, 'sentiment': 'neutral'}, {'start': 102.0, 'end': 122.89, 'text': " And when I put it into the website, it shows 2001 interchange way, but I don't know what it supplies.", 'no_speech_probability': 0.11389932781457901, 'sentiment': 'neutral'}, {'start': 122.89, 'end': 129.1, 'text': " Now, it's supposed to be an interstate battery's name?", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 129.1, 'end': 135.84, 'text': " Yes, but it's probably under NFI, which is the landlord's name.", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 135.84, 'end': 153.82, 'text': " Alright, let's see.", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 153.82, 'end': 158.71, 'text': " Three account, which means it's large power.", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 158.71, 'end': 161.71, 'text': " I can't, let me see if I can find out.", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 161.71, 'end': 167.58, 'text': ' Give me a minute.', 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 167.58, 'end': 171.96, 'text': " Did you say it's for large power?", 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 171.96, 'end': 174.14, 'text': ' Mm-hmm.', 'no_speech_probability': 0.03589588776230812, 'sentiment': 'neutral'}, {'start': 174.14, 'end': 179.14, 'text': ' Okay, that may be the upgraded service we put in.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 182.25, 'end': 184.25, 'text': ' It is, there was a work order.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 184.25, 'end': 188.53, 'text': " That's what I just got to, so let's see.", 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 188.53, 'end': 189.53, 'text': ' Give me a second.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 189.53, 'end': 195.03, 'text': " There's a work order involved that was completed.", 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 195.03, 'end': 213.49, 'text': ' Let me take a look.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 213.49, 'end': 238.9, 'text': ' The meter number you gave me, this is all.', 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 238.9, 'end': 242.06, 'text': " This is the large power, that's all I could say.", 'no_speech_probability': 0.10331965982913971, 'sentiment': 'neutral'}, {'start': 242.06, 'end': 247.06, 'text': ' I have here, it says PPL to install two more sets,', 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 249.14, 'end': 251.18, 'text': " and I don't know if this means anything,", 'no_speech_probability': 0.10594843327999115, 'sentiment': 'negative'}, {'start': 251.18, 'end': 252.94, 'text': " mate, but I don't know,", 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 252.94, 'end': 254.7, 'text': " because I don't do three-phase.", 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 254.7, 'end': 259.74, 'text': ' It says 750AL and MLP3-PTSWY.', 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 264.94, 'end': 270.3, 'text': ' So this is the large power one.', 'no_speech_probability': 0.10594843327999115, 'sentiment': 'neutral'}, {'start': 270.3, 'end': 279.1, 'text': " Okay. All right. So I need to get that put under our name then, don't I?", 'no_speech_probability': 0.2681259512901306, 'sentiment': 'neutral'}, {'start': 279.1, 'end': 284.1, 'text': " All right. Let me ask you one last question so I know where I'm going.", 'no_speech_probability': 0.2681259512901306, 'sentiment': 'neutral'}, {'start': 284.1, 'end': 291.61, 'text': " This is in a person's name. Do you know who owned the building?", 'no_speech_probability': 0.2681259512901306, 'sentiment': 'neutral'}, {'start': 291.61, 'end': 297.76, 'text': ' Uh, is it John Arnett?', 'no_speech_probability': 0.2681259512901306, 'sentiment': 'neutral'}, {'start': 297.76, 'end': 301.76, 'text': ' Perfect. Yes. Yes.', 'no_speech_probability': 0.2681259512901306, 'sentiment': 'positive'}, {'start': 301.76, 'end': 323.75, 'text': " Yeah, that's who we're dealing with, he's our landlord, he's part of the company that owns the building that we're leasing from, but I just, he emailed this morning saying,", 'no_speech_probability': 0.43267571926116943, 'sentiment': 'neutral'}, {'start': 323.75, 'end': 329.78, 'text': " Hey, we got a third meter that y'all probably need to get power", 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 329.78, 'end': 334.14, 'text': " or over y'alls name.", 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 334.14, 'end': 343.1, 'text': ' Is there any way we can backdate to January 1st', 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 343.1, 'end': 348.19, 'text': ' when the other two services went into effect?', 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 348.19, 'end': 352.5, 'text': " We're not able to go back.", 'no_speech_probability': 0.10858119279146194, 'sentiment': 'negative'}, {'start': 352.5, 'end': 354.98, 'text': ' Okay, so I need to do this.', 'no_speech_probability': 0.10858119279146194, 'sentiment': 'neutral'}, {'start': 354.98, 'end': 397.66, 'text': ' Let me tell you, this went into his name, give me a minute, so the connect was completed November 30th, so the first bill he got, give me a minute, was from November 30th to January 4th, he got one bill.', 'no_speech_probability': 0.27920442819595337, 'sentiment': 'neutral'}, {'start': 397.66, 'end': 409.92, 'text': " Anything, yeah, so, and then, now we're gonna bill, we did bill up to February 2nd, there's", 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 409.92, 'end': 421.6, 'text': " a note on here that there's billing in progress, they're billing up to, yeah, so the next", 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 421.6, 'end': 428.72, 'text': ' bill he gets would probably, I would just say between you and him, to be honest with', 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 428.72, 'end': 430.52, 'text': ' you.', 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 430.52, 'end': 431.52, 'text': ' Yeah.', 'no_speech_probability': 0.4740987718105316, 'sentiment': 'neutral'}, {'start': 431.52, 'end': 435.52, 'text': " I can't go backwards because it's already going to be long.", 'no_speech_probability': 0.31251534819602966, 'sentiment': 'negative'}, {'start': 435.52, 'end': 443.56, 'text': ' So if I want to put start service day, could I put it as today, would it go through today,', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 443.56, 'end': 446.56, 'text': ' or would it be better to put tomorrow as this?', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 446.56, 'end': 448.56, 'text': ' Tomorrow.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 448.56, 'end': 449.59, 'text': ' Okay.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 449.59, 'end': 453.59, 'text': " It's always the next day that we'll do it for.", 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 453.59, 'end': 454.59, 'text': ' Alright.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 454.59, 'end': 455.59, 'text': ' Okay.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 455.59, 'end': 456.59, 'text': " We'll do that.", 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 456.59, 'end': 457.59, 'text': ' Give me a second.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 457.59, 'end': 458.59, 'text': ' Let me just.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 458.59, 'end': 459.59, 'text': ' Full business.', 'no_speech_probability': 0.31251534819602966, 'sentiment': 'neutral'}, {'start': 459.59, 'end': 473.42, 'text': " find one of your accounts and give me a minute I can do this I don't have to do", 'no_speech_probability': 0.058932796120643616, 'sentiment': 'neutral'}, {'start': 473.42, 'end': 483.07, 'text': " it online oh no I could do it for you if you want me to it's up to you I mean I", 'no_speech_probability': 0.058932796120643616, 'sentiment': 'neutral'}, {'start': 483.07, 'end': 490.89, 'text': ' Alright, so we have, alright, are you online now?', 'no_speech_probability': 0.5168992280960083, 'sentiment': 'neutral'}, {'start': 490.89, 'end': 499.49, 'text': " Yeah, I'm, I actually, uh, point of contact, yeah.", 'no_speech_probability': 0.5168992280960083, 'sentiment': 'positive'}, {'start': 499.49, 'end': 508.5, 'text': " Yeah, you could, sure, I'll wait, make sure it goes through for ya.", 'no_speech_probability': 0.5168992280960083, 'sentiment': 'positive'}, {'start': 508.5, 'end': 514.5, 'text': " Online, honestly, wasn't gonna allow you to do it either.", 'no_speech_probability': 0.5168992280960083, 'sentiment': 'negative'}, {'start': 514.5, 'end': 533.56, 'text': " It would always give you the, it'll always give you the next business day on mine, um, but, it'll, you'll see when you get there, but, until I get to the end.", 'no_speech_probability': 0.20985403656959534, 'sentiment': 'neutral'}, {'start': 533.56, 'end': 543.56, 'text': " Actually, it's asking me for the text ID number. If, if you activate it, you can just mirror what's on the other accounts, correct?", 'no_speech_probability': 0.20985403656959534, 'sentiment': 'neutral'}, {'start': 543.56, 'end': 547.56, 'text': ' Exactly. Yes, I can, uh-huh.', 'no_speech_probability': 0.20985403656959534, 'sentiment': 'neutral'}, {'start': 547.56, 'end': 551.01, 'text': ' I will let you do that. Alright, then let me do it.', 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 551.01, 'end': 554.05, 'text': " I'll give you the account numbers. Alright, so it's", 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 554.05, 'end': 559.76, 'text': ' interstate batteries. Yep, perfect. Go ahead.', 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 559.76, 'end': 569.83, 'text': ' Okay, the... I believe this is the fire pump. It is', 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 569.83, 'end': 585.62, 'text': ' six four five seven seven dash six three zero one eight.', 'no_speech_probability': 0.19158725440502167, 'sentiment': 'neutral'}, {'start': 585.62, 'end': 588.09, 'text': ' The fire pump. Okay.', 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 588.09, 'end': 620.49, 'text': ' And meter number...', 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 620.49, 'end': 624.55, 'text': " Let's verify the meter again. Do you still have it in front of you, Jimmy?", 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 624.55, 'end': 632.39, 'text': ' Uh... let me... let me go back to it now.', 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 632.39, 'end': 644.8, 'text': ' I like to do that when I have a lot of accounts open in front of me.', 'no_speech_probability': 0.3525732457637787, 'sentiment': 'positive'}, {'start': 644.8, 'end': 647.3, 'text': " Just to make sure I'm doing the right one.", 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 647.3, 'end': 654.3, 'text': " Yes. Let me see. Where'd it go? Where'd it go? Robert, Robert.", 'no_speech_probability': 0.3525732457637787, 'sentiment': 'neutral'}, {'start': 654.3, 'end': 686.53, 'text': " No, I'm okay. I'm all right, because I just went back to John's account and took it off of there. We're okay.", 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 686.53, 'end': 687.03, 'text': ' Okay.', 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 687.03, 'end': 691.2, 'text': " Because it's the only one. We're all right.", 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 691.2, 'end': 692.2, 'text': ' Okay.', 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 692.2, 'end': 701.96, 'text': " All right, so we're doing this for tomorrow, the 13th. It's all business use.", 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 701.96, 'end': 703.44, 'text': ' Yep.', 'no_speech_probability': 0.17961890995502472, 'sentiment': 'neutral'}, {'start': 703.44, 'end': 713.54, 'text': " and just verify the mailing address for me it's in front of me so you could just verify it", 'no_speech_probability': 0.12320521473884583, 'sentiment': 'neutral'}, {'start': 717.25, 'end': 729.22, 'text': " for like invoices yes exactly correct okay it's i believe it's my address here in dallas i'm in", 'no_speech_probability': 0.12320521473884583, 'sentiment': 'neutral'}, {'start': 729.22, 'end': 738.0, 'text': " Texas. Mm-hmm. Does it? Is it? Okay. That's what I have a Texas address. Okay. Exactly.", 'no_speech_probability': 0.6127711534500122, 'sentiment': 'neutral'}, {'start': 738.0, 'end': 748.8, 'text': " Let me. All right. Let me. Let me get it. I've got so many addresses I can't remember.", 'no_speech_probability': 0.6127711534500122, 'sentiment': 'neutral'}, {'start': 750.0, 'end': 765.5, 'text': ' It is 14221 Dallas Parkway Suite 1000, Dallas, Texas 75254. Perfect. Perfect. All right.', 'no_speech_probability': 0.6127711534500122, 'sentiment': 'neutral'}, {'start': 765.74, 'end': 770.74, 'text': ' So I copied that, seven, two, five, seven, five, two.', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 771.74, 'end': 776.74, 'text': ' The zip, wait a minute, zip code is seven, five, two, five, four.', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 778.56, 'end': 781.03, 'text': " Yes, ma'am.", 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 781.03, 'end': 784.87, 'text': " All right, I didn't know if I transposed my numbers.", 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 784.87, 'end': 789.59, 'text': ' All right, just verify what phone number', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 789.59, 'end': 792.55, 'text': ' should be on the account for me.', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 792.55, 'end': 796.55, 'text': ' Two, one, four, four, four, nine, three, six, six, six.', 'no_speech_probability': 0.14805607497692108, 'sentiment': 'neutral'}, {'start': 798.46, 'end': 805.82, 'text': " Alright, and then I have your email address. We'll leave this, we'll leave it on this account also, right?", 'no_speech_probability': 0.06859400868415833, 'sentiment': 'neutral'}, {'start': 806.3, 'end': 816.77, 'text': " Yes, for now. We'll probably end up switching it. We've got an operations manager up in Quaker Town now, so.", 'no_speech_probability': 0.06859400868415833, 'sentiment': 'neutral'}, {'start': 816.77, 'end': 829.14, 'text': ' okay all right so we will start the service at 2001 interchange way in', 'no_speech_probability': 0.19277390837669373, 'sentiment': 'neutral'}, {'start': 829.14, 'end': 838.44, 'text': ' Quaker town effective tomorrow February 13th in the name of interstate', 'no_speech_probability': 0.19277390837669373, 'sentiment': 'neutral'}, {'start': 838.44, 'end': 843.9, 'text': ' batteries right now let me go through a few more things so security deposit is', 'no_speech_probability': 0.19277390837669373, 'sentiment': 'neutral'}, {'start': 843.9, 'end': 847.38, 'text': ' is not required for new service requests,', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 847.38, 'end': 852.26, 'text': ' but PPL may require a security deposit in the future', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 852.26, 'end': 856.38, 'text': ' if bill payment is received late.', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'negative'}, {'start': 856.38, 'end': 857.22, 'text': ' Okay.', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 857.22, 'end': 861.83, 'text': ' And then, give me one last thing.', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 861.83, 'end': 863.02, 'text': ' All right.', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 863.02, 'end': 867.14, 'text': ' Now, as a PPL electric utility customer,', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 867.14, 'end': 870.3, 'text': " you're entitled to certain programs and alerts,", 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 870.3, 'end': 872.62, 'text': " so I'm just gonna quickly set up the account", 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 872.62, 'end': 874.74, 'text': ' so that we can efficiently assist you', 'no_speech_probability': 0.11252597719430923, 'sentiment': 'neutral'}, {'start': 874.74, 'end': 881.04, 'text': ' any future needs. Now do you want to add an alternate phone number to the', 'no_speech_probability': 0.3322262167930603, 'sentiment': 'neutral'}, {'start': 881.04, 'end': 892.12, 'text': " account? Not yet. All right and that's okay and right now how do you prefer to", 'no_speech_probability': 0.3322262167930603, 'sentiment': 'neutral'}, {'start': 892.12, 'end': 903.36, 'text': ' be contacted? Email or not? Email. Okay all right now ask for alerts please know', 'no_speech_probability': 0.3322262167930603, 'sentiment': 'neutral'}, {'start': 903.36, 'end': 908.66, 'text': ' that alerts can be sent at any time of the day or night you will receive the', 'no_speech_probability': 0.3322262167930603, 'sentiment': 'neutral'}, {'start': 908.66, 'end': 916.26, 'text': ' my PPL alerts terms and conditions and its entirety via the US mail at the address on', 'no_speech_probability': 0.21256396174430847, 'sentiment': 'neutral'}, {'start': 916.26, 'end': 922.98, 'text': ' file for the account which is the Texas address and then you may unsubscribe to these alerts', 'no_speech_probability': 0.21256396174430847, 'sentiment': 'neutral'}, {'start': 922.98, 'end': 933.74, 'text': " to the PPL's website or by contacting PPL at 1-800-342-5775. So do I have your permission", 'no_speech_probability': 0.21256396174430847, 'sentiment': 'neutral'}, {'start': 933.74, 'end': 937.84, 'text': ' to enroll you in my PPL alerts.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 937.84, 'end': 938.84, 'text': ' Yeah.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 938.84, 'end': 939.84, 'text': ' Alrighty.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'positive'}, {'start': 939.84, 'end': 942.84, 'text': " And that'll be through email also.", 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 942.84, 'end': 944.96, 'text': ' Alright.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 944.96, 'end': 946.96, 'text': ' Now, as part of...', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 946.96, 'end': 951.96, 'text': " Oh, I see you're enrolled in paperless billing on the previous account.", 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 951.96, 'end': 955.96, 'text': " We'll continue this service on your new account.", 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 955.96, 'end': 956.96, 'text': ' Alright.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 956.96, 'end': 957.96, 'text': ' Yeah.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 957.96, 'end': 960.96, 'text': ' Um, we also offer automatic bill pay.', 'no_speech_probability': 0.17353160679340363, 'sentiment': 'neutral'}, {'start': 960.96, 'end': 965.82, 'text': ' Would you be interested in enrolling in automatic bill pay,', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 965.82, 'end': 967.82, 'text': ' where we would deduct a payment', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 967.82, 'end': 970.86, 'text': ' from a checking or savings account on the due date?', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 970.86, 'end': 974.28, 'text': ' No, not right now.', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 974.28, 'end': 976.13, 'text': ' Okay.', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 976.13, 'end': 977.83, 'text': ' All right, do you want to write down', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 977.83, 'end': 980.6, 'text': ' the new account number?', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 980.6, 'end': 997.38, 'text': ' Uh, yeah, let me,', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 997.38, 'end': 1013.88, 'text': ' let me just do that.', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 1013.88, 'end': 1018.22, 'text': ' Okay.', 'no_speech_probability': 0.02824973501265049, 'sentiment': 'neutral'}, {'start': 1018.22, 'end': 1032.88, 'text': ' The new account number is 9-4-5-8-6-dash-6-7-0-1-8.', 'no_speech_probability': 0.08544127643108368, 'sentiment': 'neutral'}, {'start': 1032.88, 'end': 1036.16, 'text': ' Oh, okay.', 'no_speech_probability': 0.08544127643108368, 'sentiment': 'neutral'}, {'start': 1036.16, 'end': 1051.05, 'text': ' And then just so you know, to make it easy for you to access your account online, we did link your PPL account to jimmy.marleratibsa.com.', 'no_speech_probability': 0.08544127643108368, 'sentiment': 'neutral'}, {'start': 1051.05, 'end': 1052.05, 'text': ' So check your email.', 'no_speech_probability': 0.08544127643108368, 'sentiment': 'neutral'}, {'start': 1052.05, 'end': 1059.78, 'text': " I'm the manager account. All right, so you are all set and have I satisfied your concerns today?", 'no_speech_probability': 0.045305266976356506, 'sentiment': 'neutral'}, {'start': 1061.78, 'end': 1071.42, 'text': ' All right, will you take care you have a good evening. Thank you. Thank you. Bye bye', 'no_speech_probability': 0.045305266976356506, 'sentiment': 'positive'}]"""

predictions = model.predict(text)

predictions

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Advanced Call Insights Llama 2

# COMMAND ----------

!pip install mlflow
!pip install ctransformers>=0.2.24
!pip3 install transformers>=4.32.0 optimum>=1.12.0
!pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7
!pip install hf-hub-ctranslate2>=2.12.0 ctranslate2>=3.17.1

# COMMAND ----------

call = """
 <PERSON>, how can I help you? Uh, yes, I was calling for a review of my bill. Well, I'd be more than happy to help with that. Can I have your name? The name <PERSON>? Um, the name on the account? I'm sorry? The name that's on the account? It's not <PERSON>? <PERSON>. Okay, and your relationship to <PERSON>? He's my husband. I just need to verify the full service address and phone number so I can get in there. <ADDRESS>. And what else did you need? Your phone number, please. <***>. Awesome. So, I see your <***> bill was <***>. The usage was [REDACTED],[REDACTED] kilowatt hours. That's roughly [REDACTED] kilowatt hours per day. It is slightly higher than <***>. <***> was <***>, but the temperatures were warmer, so that would be right in line. <***>, your bill was [REDACTED] kilowatt hours, so you're actually down from <***>. Yeah, that's what I'm calling about <***>. That was a <***> billing cycle, so it was a little bit longer. You used a total of <***>, which if you divide it by [REDACTED] is <***>. but your billed on your actual usage, not an estimate. So if you, oh, you have the online profile, you can go to our website, <URL>. You can actually look at your <***> readings and see, not that you have to sit there and look at it <***>, but you can see every day during that billing cycle what it was because i'm sure it wasn't a hundred and ten kilowatt hours <***> you used about [REDACTED] kilowatt hours per day so that was really close as well it was [REDACTED] degrees <***> [REDACTED] degrees <***> so <***> was a tiny bit warmer yeah i just I've never had a bill in the 600s, ever. I only have or had one or two in the 500s. I've never had a bill with- <***>, your bill with <***>. What? <***> was <***>. I was gonna say, I know we've had like a couple that have been like [REDACTED] but it never [REDACTED] and some yeah <***> was <***> in the billing cycle it was seven hundred five dollars seventy seven cents for the usage that was a total of <***> seventy [REDACTED] kilowatt hours <***> was <***> yeah <***> actually our rates increased to about [REDACTED] point [REDACTED] point six one two I want to say <***> they came down. Well, they went down in <***> and then they came down again in <***>. So they're still higher than they have been, but at least they're down to *** now. So that's helping a tiny bit, but it's still rough when you use a lot more electricity in <***>. I just don't <***> it. It shows like compared to <***> that we have quite a bit more usage but what would cause that because we haven't changed anything and we're doing the same thing we have the same we haven't added anything at the house we haven't done anything different what increases and how do I know that my readings are accurate Like, is there ever a time where they have to, like, check the calibration or anything on the meter? Could it be a wrong thing? Well, <***> we get a reading from your meter. When they fail, on the rare occasion that they fail, they start to slow down until they stop. And that's when we noticed, we're like, hey, why don't you use electricity anymore? Typically, they do not speed up if there's something wrong with them. These meters are extremely accurate, that's why we got them. One thing I would recommend, if you're using more electricity, you know, it's registering more usage than it did <***>. I know we had that super cold spell right towards the end of that billing period. I know that did cause a lot of usage for a lot of customers. Let me just pull up the readings real quick. Hang on. I'm just going to go over the usage dates on that bill, <***>. So let's see here, oh yeah, your usage was really high. <***>, if I could just hover and stop flashing. On <***>, your usage was *** kilowatt hours *** kilowatt hours for <***>. Temperature was [REDACTED] degrees. Towards the end of this billing cycle we had those super cold days. <***> your average temperature in your area was [REDACTED] degrees. Your usage was *** that day. So it seems like your usage is fairly high, but every time the temperatures go up, your usage goes down a little bit. Well I thought it was higher. I thought the usage was higher when it was [REDACTED] degrees. Did you say it was [REDACTED]? It was, actually. yeah why hang on wait was it <***> no i'm sorry on <***> it was <***> oh and it was [REDACTED] degrees it was where is the <***> here it is <***> it was <***> that day. So it was more. In fact, I think that is the highest day. Nope, it was higher on <***>. It was [REDACTED] degrees on <***>. So it was a couple degrees warmer, but your usage was just a tiny bit higher on <***>. So you have access to these readings. You also have access to, like I said before, you get to <***> we get a reading. So if there's something that's not functioning properly in your home, something that normally cycles on and off and is stuck on, or just something that's not as efficient as it used to be and is using more electricity, a good way to find it is by using these charts. You can turn items in your home off, either unplug them or turn them off at the breaker. If you leave them off for a decent period of time where you can get a couple readings, <***>, <***>. You can go back on the website <***>, look at the readings during that timeframe and see how much it dropped when you turn that item off. And that can help you figure out what's causing the increase in usage. Okay. Unfortunately, once the electricity goes past the meter, we can't tell where it goes. So we can't see why it's up. And I said, I mean, it's only up a tiny bit. This was about [REDACTED] kilowatt hours per day. Average temperature was <***>. <***>, yeah, it was <***>, per day but the temperature was a tiny bit colder it was average [REDACTED] so I mean there may have been a line except from [REDACTED] [REDACTED] I'm sorry <***> in <***> and it was colder by [REDACTED], the average was [REDACTED]% of [REDACTED] degrees colder <***> and then we used [REDACTED],[REDACTED] <***> and it looks like a lot when you look at the total usage but if you look at how much on you know if you divide it by <***> that's around [REDACTED] kilowatt hours per day last year this year it was at a hundred and ten so that's about [REDACTED] kilowatt hours per day if you consider that I know nobody uses the old [REDACTED] light bulbs anymore. Everybody has the LEDs. But if you had [REDACTED] [REDACTED] watt light bulbs and you ran them for <***>, that would be <***>. So the increase in usage from <***> is less than [REDACTED] light bulbs a day. So it's not really a lot. I mean, it looks like a lot, and it's disturbing anytime something goes up. But it may just be something small that's running longer than it normally does or is not cycling off. But like I said, those charts are a great way to help pinpoint it. The only other thing is if you can't find anything and you want to have your meter tested you absolutely can. Let me just see here real quick. You can send a check in for <***>. Once we receive that check we schedule a time to come out and and replace your meter. We give you a new one, we send the old one out to be tested. If it comes back within the Public Utility Commission's guidelines, it means it's not the meter and that's the end of the test. Most of the time, it almost always comes back in line. That's why we got these, because they're so accurate. But, on the off chance that it does come back outside of the Public Utility Commission's guidelines. We do refund the <***> and rebuild the account according to the test results, but because it's such a rare occurrence, I don't like to set that up as the answer to all the problems. I like to give you the other information first because if it's not the problem then it might just be a waste of time and money to have the test done and I don't want to see that happen. You know if there is something going on it's definitely gonna be true. Is that it? True. And what you would do is you would make the checkout to PPL Electric Utilities in the comment section on the check you would write meter test and your account number that way as soon as we get it we can match it up to your account and get the process rolling and on your bill let's see the older ones have the old address but the <***> bill I think that one has the new address yes post office box for one nine zero five four that's where you would send it attention meter testing okay it's the <LOCATION>. <PERSON> address oh I have an <LOCATION> address on my bill the older bills were going to <LOCATION> your <***> bill went out on <***>. That one has the <LOCATION> address on it. Okay, I haven't seen that one yet. So I haven't gotten the mail in <***> so it might be at the post office. And you said is the track gets written to PPL Electric Utilities. Correct. And then you write… And then you write… And then you write the meter testing. Uh huh. Okay. Oh, the check gets written. Yes, the check gets written to PPL Electric Utilities. In the comments section, you write your meter number… I'm sorry, your account number and meter testing. Okay. Uh huh. Okay. Okay. Well, I will talk to my husband about it and see what he wants to do and we can take a look at it too. So I can just go online on our account and watch it use that though. Exactly. Yep. Okay. Alright. I will talk to him about that and see what he wants to do. Perfect. Anything else that can help you with why you have me? me? No, that's it. Great. Does that satisfy your concerns <***>, <PERSON>? Yes, thank you. Awesome. Thanks so much for calling and for being a valued <PERSON> customer. You have a wonderful weekend. All right, you too. ***.
""" 

# COMMAND ----------

#from transformers import pipeline
#import pandas as pd
#import pandas as pd
#from ctransformers import AutoModelForCausalLM
#import requests
#import json
#import time
#from hf_hub_ctranslate2 import GeneratorCT2fromHfHub

#model_name = "michaelfeil/ct2fast-Llama-2-7b-chat-hf"
#model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
#        model_name_or_path=model_name,
#        device="cpu",
#        compute_type="int8",
        # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
#)
import ctranslate2
import functools

try:
    from transformers import AutoTokenizer
    autotokenizer_ok = True
except ImportError:
    AutoTokenizer = object
    autotokenizer_ok = False

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Any, Union, List
import os
import glob
import shutil
#from hf_hub_ctranslate2.util import utils as _utils

from typing import Optional
import huggingface_hub

from tqdm.auto import tqdm


def _download_model(
    model_name: str,
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    hub_kwargs={},
):
    """Downloads a CTranslate2 model from the Hugging Face Hub.
    Args:
      model_name: repo name on HF Hub e.g.  "michaelfeil/ct2fast-flan-alpaca-base"
      output_dir: Directory where the model should be saved. If not set,
         the model is saved in  the cache directory.
      local_files_only:  If True, avoid downloading the file and return the
        path to the local  cached file if it exists.
      cache_dir: Path to the folder where cached files are stored.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """

    kwargs = hub_kwargs
    kwargs["local_files_only"] = local_files_only
    if output_dir is not None:
        kwargs["local_dir"] = output_dir
        kwargs["local_dir_use_symlinks"] = False

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    allow_patterns = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
        "tokenizer_config.json",
        "*ocabulary.txt",
        "vocab.txt",
    ]

    return huggingface_hub.snapshot_download(
        model_name,
        allow_patterns=allow_patterns,
        tqdm_class=_disabled_tqdm,
        **kwargs,
    )


class _disabled_tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

class CTranslate2ModelfromHuggingfaceHub:
    """CTranslate2 compatibility class for Translator and Generator"""

    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cpu", "cuda"] = "cuda",
        device_index=0,
        compute_type: Literal["int8_float16", "int8"] = "int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs: dict = {},
        **kwargs: Any,
    ):
        # adaptions from https://github.com/guillaumekln/faster-whisper
        if os.path.isdir(model_name_or_path):
            model_dir = model_name_or_path
        else:
            try:
                model_dir = _download_model(
                    model_name_or_path, hub_kwargs=hub_kwargs
                )
            except Exception:
                hub_kwargs["local_files_only"] = True
                model_dir = _download_model(
                    model_name_or_path, hub_kwargs=hub_kwargs
                )


        model_bin = os.path.join(model_dir, "model.bin")
        if not os.path.exists(model_bin):
            # 
            shards = glob.glob(model_bin.replace(".bin","-.*of.*bin"))
            shards = sorted(shards, key=lambda path: int(path.split(".")[-1]))
            with open(model_bin, "wb") as model_bin_file:
                for shard in shards:
                    with open(shard, "rb") as shard_file:
                        shutil.copyfileobj(shard_file, model_bin_file)
                    os.remove(shard)
        self.model = self.ctranslate_class(
            model_dir,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            **kwargs,
        )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if "tokenizer.json" in os.listdir(model_dir):
                if not autotokenizer_ok:
                    raise ValueError(
                        "`pip install transformers` missing to load AutoTokenizer."
                    )
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, fast=True)
            else:
                raise ValueError(
                    "no suitable Tokenizer found. "
                    "Please set one via tokenizer=AutoTokenizer.from_pretrained(..) arg."
                )

    def _forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def tokenize_encode(self, text, *args, **kwargs):
        return [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(p)) for p in text
        ]

    def tokenize_decode(self, tokens_out, *args, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        text: Union[str, List[str]],
        encode_kwargs={},
        decode_kwargs={},
        *forward_args,
        **forward_kwds: Any,
    ):
        orig_type = list
        if isinstance(text, str):
            orig_type = str
            text = [text]
        token_list = self.tokenize_encode(text, **encode_kwargs)
        tokens_out = self._forward(token_list, *forward_args, **forward_kwds)
        texts_out = self.tokenize_decode(tokens_out, **decode_kwargs)
        if orig_type == str:
            return texts_out[0]
        else:
            return texts_out

class GeneratorCT2fromHfHub(CTranslate2ModelfromHuggingfaceHub):
    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cpu", "cuda"] = "cuda",
        device_index=0,
        compute_type: Literal["int8_float16", "int8"] = "int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs={},
        **kwargs: Any,
    ):
        """for ctranslate2.Generator models

        Args:
            model_name_or_path (str): _description_
            device (Literal[cpu, cuda], optional): _description_. Defaults to "cuda".
            device_index (int, optional): _description_. Defaults to 0.
            compute_type (Literal[int8_float16, int8], optional): _description_. Defaults to "int8_float16".
            tokenizer (Union[AutoTokenizer, None], optional): _description_. Defaults to None.
            hub_kwargs (dict, optional): _description_. Defaults to {}.
            **kwargs (Any, optional): Any additional arguments
        """
        self.ctranslate_class = ctranslate2.Generator
        super().__init__(
            model_name_or_path,
            device,
            device_index,
            compute_type,
            tokenizer,
            hub_kwargs,
            **kwargs,
        )

    def _forward(self, *args, **kwds):
        return self.model.generate_batch(*args, **kwds)

    def tokenize_decode(self, tokens_out, *args, **kwargs):
        return [
            self.tokenizer.decode(tokens_out[i].sequences_ids[0], *args, **kwargs)
            for i in range(len(tokens_out))
        ]

    def generate(
        self,
        text: Union[str, List[str]],
        encode_tok_kwargs={},
        decode_tok_kwargs={},
        *forward_args,
        **forward_kwds: Any,
    ):
        """_summary_

        Args:
            text (str | List[str]): Input texts
            encode_tok_kwargs (dict, optional): additional kwargs for tokenizer
            decode_tok_kwargs (dict, optional): additional kwargs for tokenizer
            max_batch_size (int, optional): _. Defaults to 0.
            batch_type (str, optional): _. Defaults to 'examples'.
            asynchronous (bool, optional): _. Defaults to False.
            beam_size (int, optional): _. Defaults to 1.
            patience (float, optional): _. Defaults to 1.
            num_hypotheses (int, optional): _. Defaults to 1.
            length_penalty (float, optional): _. Defaults to 1.
            repetition_penalty (float, optional): _. Defaults to 1.
            no_repeat_ngram_size (int, optional): _. Defaults to 0.
            disable_unk (bool, optional): _. Defaults to False.
            suppress_sequences (Optional[List[List[str]]], optional): _.
                Defaults to None.
            end_token (Optional[Union[str, List[str], List[int]]], optional): _.
                Defaults to None.
            return_end_token (bool, optional): _. Defaults to False.
            max_length (int, optional): _. Defaults to 512.
            min_length (int, optional): _. Defaults to 0.
            include_prompt_in_result (bool, optional): _. Defaults to True.
            return_scores (bool, optional): _. Defaults to False.
            return_alternatives (bool, optional): _. Defaults to False.
            min_alternative_expansion_prob (float, optional): _. Defaults to 0.
            sampling_topk (int, optional): _. Defaults to 1.
            sampling_temperature (float, optional): _. Defaults to 1.

        Returns:
            str | List[str]: text as output, if list, same len as input
        """
        return super().generate(
            text,
            encode_kwargs=encode_tok_kwargs,
            decode_kwargs=decode_tok_kwargs,
            *forward_args,
            **forward_kwds,
        )

model_name = "michaelfeil/ct2fast-Llama-2-7b-chat-hf"
model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cpu",
        compute_type="int8",
        # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
)  


# COMMAND ----------

import time

# COMMAND ----------

actions_taken_prompt_template= """
[INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue. This list should reflect the sequence of actions in the order they were taken, highlighting the agent's approach to problem-solving. Ensure the output is formatted as a list of steps, making it clear and easy to understand the progression of the agent's efforts to resolve the issue.
Keep the output limited to 300 words
<<SYS>> {} [/INST]
"""

start = time.time()
outputs = model.generate(
    text=[actions_taken_prompt_template.format(call)],
    max_length=300,
    #temperature=0.4,
    include_prompt_in_result=False
)

print(outputs[0])
print(time.time() - start)

satisfaction_score_prompt_template_v2= """
[INST] <<SYS>> Task: Assess the satisfaction level of a customer's experience based on the handling of their call by the agent, focusing on the effectiveness and efficiency of the service provided. The output should be a structured JSON-like format that includes a satisfaction score and a brief justification for the score assigned.

Instructions:
1. Evaluate the agent's handling skills during the call.
2. Assign a satisfaction score ranging from 1-10, with 1 being extremely dissatisfied and 10 being extremely satisfied.
3. Provide a 1-2 line justification for the score, highlighting key factors that influenced your rating.
4. Ensure the output is structured according to the provided satisfaction score response schema, with fields for "Score" and "Reason".
5. Format your response as a concise JSON-like output, strictly adhering to the specified structure and avoiding additional words or explanations beyond the score and its justification.

[SYS] Based on the assessment of the agent's handling skills during the call, please provide a satisfaction score and a brief justification for the score, following the structure of the satisfaction score response schema. The score should reflect the customer's level of satisfaction with the service, ranging from 1-10. Include a concise justification for the assigned score.
Conversation to be evaluated: {} [/INST]
"""

start = time.time()
outputs = model.generate(
    text=[satisfaction_score_prompt_template_v2.format(call)],
    max_length=200,
    #temperature=0.4,
    include_prompt_in_result=False
)

print(outputs[0])
print(time.time() - start)

summarization_template = """"
[INST] <<SYS>> Please summarize the key discussion points between the customer and agent in under 150 words: <<SYS>> {} [/INST]
"""

start = time.time()
outputs = model.generate(
    text=[satisfaction_score_prompt_template_v2.format(call)],
    max_length=200,
    #temperature=0.4,
    include_prompt_in_result=False
)

print(outputs[0])
print(time.time() - start)


# COMMAND ----------

class LLamainference(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the tokenizer and model for Summarization
        self.model = GeneratorCT2fromHfHub(model_name_or_path="michaelfeil/ct2fast-Llama-2-7b-chat-hf",
                                           device="cpu",
                                           compute_type="int8",
                                           # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
                                           )
        self.actions_taken_prompt_template= """
        [INST] <<SYS>> For each conversation, analyze the interaction between the customer and the agent. Identify and return a detailed list of the specific steps or actions taken by the agent to address and resolve the customer's query or issue. This list should reflect the sequence of actions in the order they were taken, highlighting the agent's approach to problem-solving. Ensure the output is formatted as a list of steps, making it clear and easy to understand the progression of the agent's efforts to resolve the issue.
        Keep the output limited to 300 words
        <<SYS>> {} [/INST]
        """
        self.satisfaction_score_prompt_template= """
        [INST] <<SYS>> Task: Assess the satisfaction level of a customer's experience based on the handling of their call by the agent, focusing on the effectiveness and efficiency of the service provided. The output should be a structured JSON-like format that includes a satisfaction score and a brief justification for the score assigned.

        Instructions:
        1. Evaluate the agent's handling skills during the call.
        2. Assign a satisfaction score ranging from 1-10, with 1 being extremely dissatisfied and 10 being extremely satisfied.
        3. Provide a 1-2 line justification for the score, highlighting key factors that influenced your rating.
        4. Ensure the output is structured according to the provided satisfaction score response schema, with fields for "Score" and "Reason".
        5. Format your response as a concise JSON-like output, strictly adhering to the specified structure and avoiding additional words or explanations beyond the score and its justification.

        [SYS] Based on the assessment of the agent's handling skills during the call, please provide a satisfaction score and a brief justification for the score, following the structure of the satisfaction score response schema. The score should reflect the customer's level of satisfaction with the service, ranging from 1-10. Include a concise justification for the assigned score.
        Conversation to be evaluated: {} [/INST]
        """

    def predict(self, context, input_text):
        steps_taken = self.model.generate(
            text=[self.actions_taken_prompt_template.format(call)],
            max_length=600,
            include_prompt_in_result=False
        )

        satisfaction_score = self.model.generate(
            text=[self.actions_taken_prompt_template.format(call)],
            max_length=600,
            include_prompt_in_result=False
        )

        return ({"Steps Taken": steps_taken[0], "Satisfaction Score":satisfaction_score[0]})

model = GeneratorCT2fromHfHub(model_name_or_path="michaelfeil/ct2fast-Llama-2-7b-chat-hf",
                              device="cpu",
                              compute_type="int8",
                              # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
                              )

model.save_pretrained("./llama_inference_pipeline")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(artifact_path="llama_inference_pipeline",
                            python_model=LLamainference(),
                            artifacts={"llama_inference_pipeline": "./llama_inference_pipeline"})


# COMMAND ----------

model_uri = "runs:/{}/llama_inference_pipeline".format(run.info.run_id)
model = mlflow.pyfunc.load_model(model_uri)

predictions = model.predict(call)

predictions


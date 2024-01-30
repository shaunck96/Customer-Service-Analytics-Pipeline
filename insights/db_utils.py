import pandas as pd
import io
from pathlib import Path
from typing import Union
import importlib.util
import sys
import tempfile
import pyarrow.feather as feather
from adlfs.spec import AzureBlobFileSystem
from pyspark.sql import SparkSession
import os
from databricks.sdk.runtime import *
import bisect
import functools
import os
import warnings
from typing import List, NamedTuple, Optional
import numpy as np
from .faster_whisper_utils import get_assets_path
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForTokenClassification

class DBUtilConnectionCreator:
    """
    Helper class for creating a connection to Azure Blob Storage using DBUtils.
    """
    def __init__(self, dbutils):
        """
        Initialize the DBUtilConnectionCreator.

        Args:
            dbutils: DBUtils instance for accessing secrets and configurations.
        """
        self.storage_key = "pplz-key-adanexpr"
        self.storage_secret = "storage-account-adanexpr"
        self.dbutils = dbutils

    def get_abfs_client(self):
        """
        Create an Azure Blob Storage client for working with Azure Blob Storage

        Returns:
            AzureBlobFileSystem: The Azure Blob Storage client.
        """
        try:
            spark = SparkSession.builder.getOrCreate()

            ppl_tenant_id = self.dbutils.secrets.get(
                scope=self.storage_key,
                key="tenant-id-adanexpr")
            adanexpr_storage_acct = self.dbutils.secrets.get(
                scope="pplz-key-adanexpr",
                key="storage-account-adanexpr")
            adanexpr_ds_dbricks_id = self.dbutils.secrets.get(
                scope="pplz-key-adanexpr",
                key="Azure-SP-ADANEXPR-DS-DBricks-ID")
            adanexpr_ds_dbricks_pwd = self.dbutils.secrets.get(
                scope="pplz-key-adanexpr",
                key="Azure-SP-ADANEXPR-DS-DBricks-PWD")

            spark.conf.set("fs.azure.enable.check.access", "false")

            acct = adanexpr_storage_acct
            config_key = f"fs.azure.account.auth.type.{acct}.dfs.\
                core.windows.net"
            config_value = "OAuth"
            spark.conf.set(config_key, config_value)

            config_key = f"fs.azure.account.hns.enabled.{acct}.dfs.\
                core.windows.net"
            config_value = "true"
            spark.conf.set(config_key, config_value)

            config_key = f"fs.azure.account.oauth.provider.type.{acct}.dfs.\
                core.windows.net"
            config_value = "org.apache.hadoop.fs.azurebfs.oauth2.\
                ClientCredsTokenProvider"
            spark.conf.set(config_key, config_value)

            config_key = f"fs.azure.account.oauth2.client.id.{acct}.\
                dfs.core.windows.net"
            config_value = adanexpr_ds_dbricks_id
            spark.conf.set(config_key, config_value)

            config_key_secret = ("fs.azure.account.oauth2."
                                 f"client.secret.{acct}."
                                 "dfs.core.windows.net")
            config_value_secret = adanexpr_ds_dbricks_pwd
            spark.conf.set(config_key_secret, config_value_secret)

            endpoint = (f"https://login.microsoftonline.com/{ppl_tenant_id}/"
                        "oauth2/token")
            cke = (f"fs.azure.account.oauth2.client.endpoint.{acct}."
                   "dfs.core.windows.net")
            spark.conf.set(cke, endpoint)

            return AzureBlobFileSystem(account_name=adanexpr_storage_acct,
                                       tenant_id=ppl_tenant_id,
                                       client_id=adanexpr_ds_dbricks_id,
                                       client_secret=adanexpr_ds_dbricks_pwd)
        except Exception as e:
            print(f"Error in get_abfs_client: {str(e)}")
            return None

    def write_df_to_azure(self,
                          abfs_client,
                          input_file: pd.DataFrame,
                          azure_path: Union[str, Path],
                          format="feather", verbose=True):
        """
        Write a DataFrame to Azure Blob Storage.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            input_file (pd.DataFrame): The DataFrame to be written.
            azure_path (Union[str, Path]): The path in Azure Blob
            Storage where the DataFrame will be stored.
            format (str): The format in which to
            store the DataFrame (feather, csv, or parquet).
            verbose (bool): Whether to print a message after writing the file.
        """
        try:
            stream_file = io.BytesIO()
            if format == "feather":
                feather.write_feather(input_file, stream_file)
            elif format == "csv":
                input_file.to_csv(stream_file)
            else:
                input_file.to_parquet(stream_file)
            file_to_blob = stream_file.getvalue()

            with abfs_client.open(azure_path, "wb") as file_obj:
                file_obj.write(file_to_blob)
            if verbose:
                print(f"File written: {str(azure_path)}")
        except Exception as e:
            print(f"Error in write_df_to_azure: {str(e)}")
    
    def read_df_from_azure(self,
                           abfs_client,
                           azure_path: Union[str, Path],
                           format="feather", verbose=True) -> pd.DataFrame:
        """
        Read a DataFrame from Azure Blob Storage.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            azure_path (Union[str, Path]): The path in Azure Blob
            Storage where the DataFrame is stored.
            format (str): The format in which the DataFrame is stored (feather, csv, or parquet).
            verbose (bool): Whether to print a message after reading the file.

        Returns:
            pd.DataFrame: The DataFrame read from Azure Blob Storage.
        """
        try:
            with abfs_client.open(azure_path, "rb") as file_obj:
                stream_file = io.BytesIO(file_obj.read())

            if format == "feather":
                df = feather.read_feather(stream_file)
            elif format == "csv":
                df = pd.read_csv(stream_file)
            else:
                df = pd.read_parquet(stream_file)

            if verbose:
                print(f"File read: {str(azure_path)}")
            return df
        except Exception as e:
            print(f"Error in read_df_from_azure: {str(e)}")
            return None
        
    def clear_blob_directory(self, abfs_client, directory_path):
        """
        Clear all files in a specified Azure Blob directory.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            directory_path (str): The path to the directory in Azure Blob Storage.
        """
        try:
            for file_path in abfs_client.ls(directory_path):
                abfs_client.rm(file_path)
            print(f"Directory cleared: {directory_path}")
        except Exception as e:
            print(f"Error in clear_blob_directory: {str(e)}")

    def count_files_in_directory(self, abfs_client, directory_path):
        """
        Count the number of files in a specified Azure Blob directory.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            directory_path (str): The path to the directory in Azure Blob Storage.

        Returns:
            int: The number of files in the directory.
        """
        try:
            return len(abfs_client.ls(directory_path))
        except Exception as e:
            print(f"Error in count_files_in_directory: {str(e)}")
            return 0

    def duplicate_blob_folder(self, abfs_client, source_directory, target_directory):
        """
        Create a duplicate of a specified blob folder within the same directory.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            source_directory (str): The path to the source directory in Azure Blob Storage.
            target_directory (str): The path to the target directory in Azure Blob Storage.
        """
        try:
            for file_path in abfs_client.ls(source_directory):
                file_name = file_path.split('/')[-1]
                abfs_client.cp(file_path, f"{target_directory}/{file_name}")
            print(f"Folder duplicated from {source_directory} to {target_directory}")
        except Exception as e:
            print(f"Error in duplicate_blob_folder: {str(e)}")

    def copy_file(self, abfs_client, source_file_path, target_file_path):
        """
        Copy a file from one folder to another in Azure Blob Storage.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            source_file_path (str): The path to the source file in Azure Blob Storage.
            target_file_path (str): The path to the target file in Azure Blob Storage.
        """
        try:
            abfs_client.cp(source_file_path, target_file_path)
            print(f"File copied from {source_file_path} to {target_file_path}")
        except Exception as e:
            print(f"Error in copy_file: {str(e)}")

    def list_directory_contents(self, abfs_client, directory_path):
        """
        List all file and folder names in a specified Azure Blob directory.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            directory_path (str): The path to the directory in Azure Blob Storage.

        Returns:
            list: A list of names of all files and folders in the directory.
        """
        try:
            contents = abfs_client.ls(directory_path)
            return [item.split('/')[-1] for item in contents]
        except Exception as e:
            print(f"Error in list_directory_contents: {str(e)}")
            return []

    def get_latest_upload_timestamp(self, abfs_client, directory_path):
        """
        Get the timestamp of the latest upload to a specified Azure Blob directory.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            directory_path (str): The path to the directory in Azure Blob Storage.

        Returns:
            datetime: The timestamp of the most recent upload, or None if the directory is empty.
        """
        try:
            file_paths = abfs_client.ls(directory_path)
            if not file_paths:
                return None

            latest_timestamp = None
            for file_path in file_paths:
                file_info = abfs_client.info(file_path)
                if 'modificationTime' in file_info:
                    timestamp = file_info['modificationTime']
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
            
            if latest_timestamp is not None:
                return datetime.datetime.fromtimestamp(latest_timestamp / 1000)
            else:
                return None
        except Exception as e:
            print(f"Error in get_latest_upload_timestamp: {str(e)}")
            return None

    def calculate_directory_size(self, abfs_client, directory_path):
        """
        Calculate the total size occupied by files in a specified Azure Blob directory.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            directory_path (str): The path to the directory in Azure Blob Storage.

        Returns:
            int: The total size of all files in the directory in bytes, or 0 if the directory is empty.
        """
        try:
            total_size = 0
            items = abfs_client.ls(directory_path, detail=True)

            for item in items:
                # Check if the item is a file (not a directory)
                if item.get('isDirectory', False):
                    continue

                # Accumulate the size
                total_size += item.get('length', 0)

            return total_size
        except Exception as e:
            print(f"Error in calculate_directory_size: {str(e)}")
            return 0

    def create_folder(self, abfs_client, folder_path):
        """
        Create a folder in a specified Azure Blob Storage directory.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            folder_path (str): The path to the folder to be created in Azure Blob Storage.
        """
        try:
            # Check if the folder already exists
            if not abfs_client.exists(folder_path):
                # The folder doesn't exist, so create it.
                # Azure Blob Storage doesn't have a dedicated 'create folder' method.
                # Folders are virtual and are created automatically when a file is created.
                # Hence, we create an empty file to establish the folder.
                dummy_file_path = f"{folder_path}/dummy.txt"
                with abfs_client.open(dummy_file_path, "wb") as file_obj:
                    file_obj.write(b'')
                # Remove the dummy file after creating the folder
                abfs_client.rm(dummy_file_path)

                print(f"Folder created: {folder_path}")
            else:
                print(f"Folder already exists: {folder_path}")
        except Exception as e:
            print(f"Error in create_folder: {str(e)}")

    def download_and_load_gramformer_model(self, abfs_client, blob_directory):
        """
        Download Hugging Face model files from Azure Blob Storage to a temporary directory,
        and load the model and tokenizer.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            blob_directory (str): Blob directory path where model files are stored.

        Returns:
            model: The loaded Hugging Face model.
            tokenizer: The loaded tokenizer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # List all blobs in the directory and download them
                blob_list = abfs_client.ls(blob_directory)
                for blob in blob_list:
                    blob_name = os.path.basename(blob)  # Extract just the file name
                    blob_path = os.path.join(blob_directory, blob_name)
                    local_file_path = os.path.join(temp_dir, blob_name)

                    with abfs_client.open(blob_path, "rb") as remote_file:
                        with open(local_file_path, "wb") as local_file:
                            local_file.write(remote_file.read())

                print(f"All model files downloaded to temporary directory")

                # Load the model and tokenizer from the temporary directory
                tokenizer = AutoTokenizer.from_pretrained(temp_dir) 
                model = AutoModelForSeq2SeqLM.from_pretrained(temp_dir) 
                print("Model and tokenizer loaded successfully.")
                return model, tokenizer

            except Exception as e:
                print(f"Error in download_and_load_model: {str(e)}")
                return None, None

    def download_and_load_redaction_model(self, abfs_client, blob_directory):
        """
        Download Hugging Face model files from Azure Blob Storage to a temporary directory,
        and load the model and tokenizer.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            blob_directory (str): Blob directory path where model files are stored.

        Returns:
            model: The loaded Hugging Face model.
            tokenizer: The loaded tokenizer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # List all blobs in the directory and download them
                blob_list = abfs_client.ls(blob_directory)
                for blob in blob_list:
                    blob_name = os.path.basename(blob)  # Extract just the file name
                    blob_path = os.path.join(blob_directory, blob_name)
                    local_file_path = os.path.join(temp_dir, blob_name)

                    with abfs_client.open(blob_path, "rb") as remote_file:
                        with open(local_file_path, "wb") as local_file:
                            local_file.write(remote_file.read())

                print(f"All model files downloaded to temporary directory")

                # Load the model and tokenizer from the temporary directory
                tokenizer = AutoTokenizer.from_pretrained(temp_dir)
                model = AutoModelForTokenClassification.from_pretrained(temp_dir)
                print("Model and tokenizer loaded successfully.")
                return model, tokenizer

            except Exception as e:
                print(f"Error in download_and_load_model: {str(e)}")
                return None, None

    def download_and_load_finetuned_t5_summarizer(self, abfs_client, blob_directory):
        """
        Download Hugging Face model files from Azure Blob Storage to a temporary directory,
        and load the model and tokenizer.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            blob_directory (str): Blob directory path where model files are stored.

        Returns:
            model: The loaded Hugging Face model.
            tokenizer: The loaded tokenizer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # List all blobs in the directory and download them
                blob_list = abfs_client.ls(blob_directory)
                for blob in blob_list:
                    blob_name = os.path.basename(blob)  # Extract just the file name
                    blob_path = os.path.join(blob_directory, blob_name)
                    local_file_path = os.path.join(temp_dir, blob_name)

                    with abfs_client.open(blob_path, "rb") as remote_file:
                        with open(local_file_path, "wb") as local_file:
                            local_file.write(remote_file.read())

                print(f"All model files downloaded to temporary directory")

                # Load the model and tokenizer from the temporary directory
                tokenizer = AutoTokenizer.from_pretrained(temp_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(temp_dir)
                print("Model and tokenizer loaded successfully.")
                return model, tokenizer

            except Exception as e:
                print(f"Error in download_and_load_model: {str(e)}")
                return None, None

    def download_and_load_zeroshot_model(self, abfs_client, blob_directory):
        """
        Download Hugging Face model files from Azure Blob Storage to a temporary directory,
        and load the model and tokenizer.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            blob_directory (str): Blob directory path where model files are stored.

        Returns:
            model: The loaded Hugging Face model.
            tokenizer: The loaded tokenizer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # List all blobs in the directory and download them
                blob_list = abfs_client.ls(blob_directory)
                for blob in blob_list:
                    blob_name = os.path.basename(blob)  # Extract just the file name
                    blob_path = os.path.join(blob_directory, blob_name)
                    local_file_path = os.path.join(temp_dir, blob_name)

                    with abfs_client.open(blob_path, "rb") as remote_file:
                        with open(local_file_path, "wb") as local_file:
                            local_file.write(remote_file.read())

                print(f"All model files downloaded to temporary directory")

                # Load the model and tokenizer from the temporary directory
                classifier = pipeline("zero-shot-classification", model=temp_dir)
                print("Model and tokenizer loaded successfully.")
                return classifier

            except Exception as e:
                print(f"Error in download_and_load_model: {str(e)}")
                return None, None

    def import_python_file_from_blob(self, abfs_client, blob_directory, file_name):
        """
        Import a Python file from Azure Blob Storage as a module.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            blob_directory (str): The directory in Azure Blob Storage where the Python file is located.
            file_name (str): The name of the Python file to be imported.

        Returns:
            module: The imported Python module.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                blob_path = os.path.join(blob_directory, file_name)
                local_file_path = os.path.join(temp_dir, file_name)

                # Download the Python file
                with abfs_client.open(blob_path, "rb") as remote_file:
                    with open(local_file_path, "wb") as local_file:
                        local_file.write(remote_file.read())

                print(f"Python file '{file_name}' downloaded to temporary directory")

                # Load the Python file as a module
                spec = importlib.util.spec_from_file_location(file_name, local_file_path)
                python_module = importlib.util.module_from_spec(spec)
                sys.modules[file_name] = python_module
                spec.loader.exec_module(python_module)
                
                print(f"Module '{file_name}' imported successfully.")
                return python_module

            except Exception as e:
                print(f"Error in import_python_file_from_blob: {str(e)}")
                return None

    def create_text_file(self, abfs_client, directory_path, file_name, content):
        """
        Create a text file in a specified Azure Blob directory with the given content.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            directory_path (str): The path to the directory in Azure Blob Storage.
            file_name (str): The name of the text file to be created.
            content (str): The content to be written to the text file.
        """
        try:
            # Create the full path to the text file
            file_path = f"{directory_path}/{file_name}"

            # Write the content to the text file
            with abfs_client.open(file_path, "wb") as file_obj:
                file_obj.write(content.encode("utf-8"))

            print(f"Text file '{file_name}' created in directory '{directory_path}'")
        except Exception as e:
            print(f"Error in create_text_file: {str(e)}")

    def download_and_load_sentiment_model(self, abfs_client, blob_directory):
        """
        Download Hugging Face model files from Azure Blob Storage to a temporary directory,
        and load the pipeline.

        Args:
            abfs_client (AzureBlobFileSystem): The Azure Blob Storage client.
            blob_directory (str): Blob directory path where model files are stored.

        Returns:
            pipe: The loaded Hugging Face pipeline with model and tokenizer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # List all blobs in the directory and download them
                blob_list = abfs_client.ls(blob_directory)
                for blob in blob_list:
                    blob_name = os.path.basename(blob)  # Extract just the file name
                    blob_path = os.path.join(blob_directory, blob_name)
                    local_file_path = os.path.join(temp_dir, blob_name)

                    with abfs_client.open(blob_path, "rb") as remote_file:
                        with open(local_file_path, "wb") as local_file:
                            local_file.write(remote_file.read())

                print("All model files downloaded to temporary directory")

                # Load the pipeline from the temporary directory
                pipe = pipeline("sentiment-analysis", model=temp_dir, tokenizer=temp_dir)
                print("Pipeline loaded successfully.")
                return pipe
            except Exception as e:
                print(f"An error occurred: {e}")
                return None

    def download_and_load_onnx_runtime(self, abfs_client):
        """
        Download and load ONNX runtime model for VAD detection
        """
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4

        model_path = 'datascience/data/ds/sandbox/shibushaun/huggingface_models/assets/silero_vad.onnx'

        with abfs_client.open(model_path, 'rb') as f:
            model_bytes = f.read()
        
        session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"], sess_options=opts)

        return session


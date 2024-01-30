from setuptools import setup, find_packages
import os

# Assuming that the package name is `cspanlp` and the ONNX file is in `cspanlp/assets`
PACKAGE_NAME = 'cspanlp'  # replace with your actual package name

def load_requirements(*file_paths):
    requirements = []
    for file_path in file_paths:
        abs_file_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(abs_file_path, 'r') as file:
            requirements.extend(file.read().splitlines())
    return requirements

requirements_files = [
    'cs_requirements.txt',
    'faster_whisper_requirements.txt',
    'gramformer_requirements.txt',
    'pii_redaction_requirements.txt'
]

long_description = ''
with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME,
    version='0.1',
    packages=find_packages(),
    description='A package for NLP processing of customer service call data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Shaun Shibu',
    author_email='sshibu@pplweb.com',
    install_requires=load_requirements(*requirements_files),
    python_requires='>=3.6',
    scripts=[],
)

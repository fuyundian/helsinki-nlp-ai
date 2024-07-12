from setuptools import setup, find_packages

setup(
    name='helsinki-nlp-ai',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        "transformers",
        "torch",
        "sentencepiece",
        "sacremoses",
        "gunicorn",
        "Flask",
        "uvicorn",
        "fastapi"
    ],
    entry_points={
        'console_scripts': [
        ],
    },
)

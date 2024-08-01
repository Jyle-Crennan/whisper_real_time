import setuptools

setuptools.setup(
    name='ASL_STT_and_NLP',
    version='1.0',
    description='Speech-to-text and NLP that uses the whisper_real_time STT model',
    author='Kyle-Brennan',
    install_requirements=[
        'pyaudio',
        'SpeechRecognition',
        '--extra-index-url https://download.pytorch.org/whl/cu116',
        'torch~=2.3.1',
        'numpy~=1.26.4',
        'git+https://github.com/openai/whisper.git',
        'nltk~=3.8.1',
        'contractions~=0.1.73',
        'spacy~=3.7.2',
        'keyboard~=0.13.5'
    ]
)

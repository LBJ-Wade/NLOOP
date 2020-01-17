from setuptools import setup

with open("requirements.txt", "r") as f:
    reqs = [line.rstrip("\n") for line in f if line != "\n"]

setup(
    name='NLOOP',
    version='v0.0',
    packages=['nloop', 'nloop.lib'],
    install_requires=reqs,
    url='https://github.com/syasini/NLOOP',
    license='MIT',
    author='Siavash Yasini, Amin Oji',
    author_email='syasini@dexm.com, aoji@dexm.com',
    description='A python package for conveniently processing, analyzing, and modeling text data'
    )

# make sure the spacy 'en' library is installed
import spacy
try:
    spacy.load('en')
except OSError:
    print("Spacy model 'en' is not installed. Downloading it now...")
    from spacy.cli import download
    download('en')
    print("Done")

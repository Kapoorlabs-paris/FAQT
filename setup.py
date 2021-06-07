import setuptools
from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name="faqt",
    version='0.0.1',

    author='Varun Kapoor,Blanco Obregon Dalmiro,Laura Boulan',
    author_email='randomaccessiblekapoor@gmail.com',
    url='https://github.com/kapoorlab/faqt/',
    description='Flywing asymmetry quantification tool.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "numpy",
        "pandas",
        "csbdeep"
        "scikit-image",
        "scipy",
        "tifffile",
        "matplotlib",
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)

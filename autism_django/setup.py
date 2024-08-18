from setuptools import find_packages , setup

from typing import List


# Declaring Variables for setup functions
PROJECT_NAME="Autism Prediction"
VERSION = "0.0.1"
AUTHOR = 'Bhavesh'
DESCRIPTION = "This is Austism Prediction"
PACKAGES= find_packages()
REQUIREMENT_FILE_NAME= "requirements.txt"


def get_requirements()->List[str]:
    """
    Description : This function is going to return list of requirement mention in requirements.txt file

    return this function is going to return a list which
    contain name of libraries mentioned in requirements.txt file

    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        print(requirement_file.readlines())
        return requirement_file.readlines().remove('-e .\n')



setup(
    name=PROJECT_NAME,
    version=VERSION,
    author =AUTHOR,
    description= DESCRIPTION,
    packages=PACKAGES,
    install_requires=get_requirements()
)
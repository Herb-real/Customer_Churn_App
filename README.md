# Customer Churn App with Gradio
_________________________________________________________________________________________________

# Introduction
_________________________________________________________________________________________________
Gradio represents a Python library that expeditiously enables the creation of adaptable UI (User Interface) components for your machine learning models, neural networks, and other functionalities. By utilizing Gradio, generating a web app or desktop app for your machine learning model becomes a straightforward task. It emerges as a valuable asset for data scientists and machine learning engineers seeking swift model building, testing, and deployment capabilities.

We’re set to build a customer churn prediction application using Gradio. Our approach involves employing customer data sourced from a telecommunications company to train a machine learning model. Subsequently, we will craft a Gradio application that ingests customer data and produces predictions regarding their likelihood of churning.

# Setup
_________________________________________________________________________________________________
Install the required packages to be able to run the evaluation locally.

You need to have Python3 on your system. Then you can clone this repo and being at the repo's root (root :: repo_name> ...) follow the steps below:

 - Windows:
python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt

 - Linux & MacOs:
python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt

The both long command-lines have a same structure, they pipe multiple commands using the symbol ; but you may manually execute them one after another.

1. Create the Python's virtual environment that isolates the required libraries of the project to avoid conflicts;
2. Activate the Python's virtual environment so that the Python kernel & libraries will be those of the isolated environment;
3. Upgrade Pip, the installed libraries/packages manager to have the up-to-date version that will work correctly;
4. Install the required libraries/packages listed in the requirements.txt file so that it will be allow to import them into the   python's scripts and notebooks without any issue.

NB: For MacOs users, please install Xcode if you have an issue.

# 🔧 The Gradio App
__________________________________________________________________________________________________
We intend to utilize the Gradio library for app development. Within the app, there will be interactive elements like dropdown menus, sliders, and radio buttons that collect user inputs and retain them as variables. Additionally, we will establish an output element that presents the churn prediction produced by the machine learning model.

# 🚀 Execution
_________________________________________________________________________________________________
To Run the app: Start the Gradio app by calling the launch() method on the Gradio interface object.

block.launch()

This will launch a local web server and open a browser window with the Gradio interface. Then you just click on run and you're good to go.

# Screenshot
_________________________________________________________________________________________________
![Alt text](<Screenshot (183).png>) 
![Alt text](<Screenshot (184).png>)

# Author
Herbert DJANIE

# Masters Thesis

This repo contains the code required for the data extraction, labelling
and finally the architecture required in order to create a subcategory discrimination classifier.
This architecture was built with decentralized social media instances in mind and to be able to run on low resources.


# Use 

The use of a venv is highly recommended. A virtual environment can be created with the following command:
```
python3 -m venv venv 
```
and activated with following command:
```
source venv/bin/activate
```
if you're on windows, on powershell execute to activate the virtual env:
```
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\venv\Scripts\Activate.ps1
```
after the venv is installed, run the following to download all packages:
```
pip install -r requirements.txt
```
please create a .venv file too and paste your HuggingFace API key within it as such:
```
HF_TOKEN=KEY
```





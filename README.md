# EEG Space

###
This software has the goal reading in EEG data from [Spike2 scripts](https://ced.co.uk/downloads/scriptspkedit)
The software has two parts, the first is offline analysis which can be used to create a model for your eeg brain data
This is an interactive iterative process, where you provide input files play around with the clustering and representation to create your model and then you can create the model
The model is then used by the online analysis which will take live spike data taken from output files and will give you a categorization of the signal based on your model. 

### How to run:

1. Install python 3.9 as this is the only version compatable with sonpy, sonpy is used to read in Spike2 script files
2. Create a virtual environment using python - venv <your venv name>
3. Activate the virtual environment, on windows this is done with <your venv name>\Scripts\Activate.ps1
4. Install the requiements from the requirements.txt file in the root directory via pip install -r .\requirements.txt
5. Run the software with python Main.py

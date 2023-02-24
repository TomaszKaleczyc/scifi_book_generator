ENV_FOLDER=./environment
VENV_NAME=___venv
VENV_PATH=$(ENV_FOLDER)/$(VENV_NAME)
VENV_ACTIVATE_PATH=$(VENV_PATH)/bin/activate
REQUIREMENTS_PATH=$(ENV_FOLDER)/requirements.txt
PYTHON_VERSION = python3.8


help:
	@echo "======================== Project makefile  ========================"
	@echo "Use make {command} with any of the below options:"
	@echo "    * create_env - creating the project virtual environment"
	@echo "    * activate-env-command - printing the command to activate environment in the console"

create-env:
	@echo "======================== Creating the project virtual environment ========================" 
	$(PYTHON_VERSION) -m virtualenv --system-site-packages -p $(PYTHON_VERSION) $(VENV_PATH)
	. $(VENV_ACTIVATE_PATH) && \
	$(PYTHON_VERSION) -m pip install pip --upgrade && \
	$(PYTHON_VERSION) -m pip install --upgrade six && \
	$(PYTHON_VERSION) -m pip install -r $(REQUIREMENTS_PATH)

activate-env-command:
	@echo "======================== Execute the below command in terminal ========================" 
	@echo source $(VENV_ACTIVATE_PATH)

download-dataset:
	. $(VENV_ACTIVATE_PATH) && \
	cd data && \
	kaggle datasets download jannesklaas/scifi-stories-text-corpus && \
	unzip scifi-stories-text-corpus.zip && \
	rm scifi-stories-text-corpus.zip

purge-output:
	rm -r output/*

run-training:
	. $(VENV_ACTIVATE_PATH) && \
	cd src/ && \
	python train.py

run-test:
	. $(VENV_ACTIVATE_PATH) && \
	cd src/ && \
	python test.py
.PHONY: format
format:
	black ./continuous-umps

.PHONY: style
style:
	flake8 ./continuous-umps

.PHONY: test
test:
	pytest ./continuous-umps/tests/

.PHONY: train
train:
	python3 ./continuous-umps/train_scripts.py

.PHONY: format
format:
	black ./

.PHONY: style
style:
	flake8

.PHONY: test
test:
	pytest ./tests

PYTHON ?= python3
JOBS ?= 12

.PHONY: engine test baseline merge validate report clean

engine:
	$(MAKE) -C engine

test: engine
	$(PYTHON) -m unittest discover -s tests

baseline:
	$(PYTHON) -m elfcalc solve baseline --jobs $(JOBS)
	$(PYTHON) -m elfcalc solve baseline --practice --jobs $(JOBS)

merge:
	$(PYTHON) -m elfcalc merge

validate:
	$(PYTHON) -m elfcalc validate submissions/submission.csv

report:
	$(PYTHON) -m elfcalc report --practice
	$(PYTHON) -m elfcalc report

clean:
	$(MAKE) -C engine clean

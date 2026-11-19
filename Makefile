PYTHON ?= python3
JOBS ?= 12

.PHONY: engine test baseline solve-practice solve-scored merge validate report clean

engine:
	$(MAKE) -C engine

test: engine
	$(PYTHON) -m unittest discover -s tests

baseline:
	$(PYTHON) -m elfcalc solve baseline --jobs $(JOBS)
	$(PYTHON) -m elfcalc solve baseline --practice --jobs $(JOBS)

solve-practice: engine
	$(PYTHON) -m elfcalc solve baseline --practice --jobs $(JOBS)
	$(PYTHON) -m elfcalc solve exact --practice --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve greedy --practice --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve greedy_v2 --practice --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve mitm_basis --practice --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve shared_dag --practice --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve mult_factor --practice --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve mixed_hybrid --practice --jobs $(JOBS) || true

solve-scored: engine
	$(PYTHON) -m elfcalc solve baseline --jobs $(JOBS)
	$(PYTHON) -m elfcalc solve exact --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve greedy --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve greedy_v2 --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve mitm_basis --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve shared_dag --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve mult_factor --jobs $(JOBS) || true
	$(PYTHON) -m elfcalc solve mixed_hybrid --jobs $(JOBS) || true

merge:
	$(PYTHON) -m elfcalc merge

validate:
	$(PYTHON) -m elfcalc validate submissions/submission.csv

report:
	$(PYTHON) -m elfcalc report --practice
	$(PYTHON) -m elfcalc report

clean:
	$(MAKE) -C engine clean

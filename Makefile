.PHONY: demo test oracle clean

demo:
	python3 dse.py --iters 300

oracle:
	python3 dse.py --oracle

test:
	python3 test_dse.py

clean:
	rm -rf build __pycache__

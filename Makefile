driver_path:
	which chromedriver > txt/driver_path.txt

show_keywords:
	cat txt/keywords.txt

cookie:
	python -B src/CAPTCHA.py
	python -B src/applyCookies.py

search_scholar:
	python -B src/ExtractGoogleScholar.py

search_arxiv:
	python -B src/arxiv.py

search_MLM:
	python -B src/MachineLearningMastery.py

search_medium:
	python -B src/medium.py
	
search_youtube:
	python -B src/Youtube.py

test:
	python -B src/test.py
	
test_iter:
	python -B src/test_iteration.py

clean:
	rm data/*
	rm temp/*
	rm txt/output_*
	rm -rf fig/*
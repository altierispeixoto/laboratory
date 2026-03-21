
scrapy runspider src/extractor/trademe.py -o trademe.json

scrapy crawl trademe -o data/raw/trademe-2023-01-17.jsonl -L WARNING
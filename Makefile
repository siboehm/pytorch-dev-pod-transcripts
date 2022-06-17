episodes.md: deepgram_transcribe.py
	python deepgram_transcribe.py

README.md: episodes.md _README.md
	cat _README.md episodes.md > README.md
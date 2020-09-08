mkdir -p ./input/tweet-sentiment-extraction
cd input/tweet-sentiment-extraction || exit
kaggle competitions download -c tweet-sentiment-extraction
unzip tweet-sentiment-extraction.zip
rm tweet-sentiment-extraction.zip
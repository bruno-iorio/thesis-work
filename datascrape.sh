## embeddings
wget -P ./data/ https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip data/wiki-news-300d-1M.vec.zip
rm data/wiki-news-300d-1M.vec.zip

## MELD dataset
wget -P ./data/ https://raw.githubusercontent.com/declare-lab/MELD/refs/heads/master/data/MELD/dev_sent_emo.csv
wget -P ./data/ https://raw.githubusercontent.com/declare-lab/MELD/refs/heads/master/data/MELD/test_sent_emo.csv
wget -P ./data/ https://raw.githubusercontent.com/declare-lab/MELD/refs/heads/master/data/MELD/train_sent_emo.csv

### TL;DR
This is a brief tutorial to inform you how to build a search engine for wikipedia using ElasticSearch.

### Install ElasticSearch
You can download and install elasticsearch from [here](https://www.elastic.co/downloads/elasticsearch). ElasticSearch 7.17 is used here.
Some plugins are also needed for this project:
- [analysis-icu](https://www.elastic.co/guide/en/elasticsearch/plugins/7.17/analysis-icu.html) (for any language)
- [analysis-stconvert](https://github.com/medcl/elasticsearch-analysis-stconvert) (for Chinese)

### Download wikipedia
Wiki dump for [English](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2) and for [Chinese](https://dumps.wikimedia.org/zhwiki/latest/enwiki-latest-pages-articles.xml.bz2) is offered here. Dumps for another language/ another wiki database should be similar.

After a wiki dump downloaded, moving it into the `wiki_dump` folder under this project (create one in case you can't find it).

### Extract wiki text
[wikiextractor](https://github.com/attardi/wikiextractor) is used here to extract wiki text.

Please follow the tutorial in [wikiextractor](https://github.com/attardi/wikiextractor) to install the tool.

When finished, just extract wiki text by `wikiextractor wiki_dump/$wikidata -o wiki_dump/extracted_$wikidata --json`, which will extract and save file in json format.


### Import setting and mapping files
Run the following command for wiki dump for English.

An index named `enwiki` will be initialized.
```bash
curl -XDELETE localhost:9200/enwiki?pretty
cat ./config/settings_en.json | curl -H 'Content-Type: application/json' -XPUT localhost:9200/enwiki?pretty -d @-
cat ./config/mappings.json | curl -H 'Content-Type: application/json' -XPUT localhost:9200/enwiki/_mapping?pretty -d @-
```

Run the following command for wiki dump for Chinese.

An index named `zhwiki` will be initialized.
```bash
curl -XDELETE localhost:9200/zhwiki?pretty
cat ./config/settings_zh.json | curl -H 'Content-Type: application/json' -XPUT localhost:9200/zhwiki?pretty -d @-
cat ./config/mappings.json | curl -H 'Content-Type: application/json' -XPUT localhost:9200/zhwiki/_mapping?pretty -d @-
```

### Build index
Run the following command and index will be built. Remember to modify `index_name` to use specific index.
```python
python build_index.py
```


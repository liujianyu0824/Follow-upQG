# Follow-up Question Generation with Knowledge Graph and LLM

## Overview
This project implements a novel approach for generating follow-up questions by integrating external knowledge from Wikipedia with large language models (LLMs). This research has been accepted by the 31st International Conference on Computational Linguistics(COLING 2025). The method follows a three-stage process:

1. **Recognition Module**: Identifies key topics from the conversation history and retrieves relevant Wikipedia pages.
2. **Selection Module**: Constructs a real-time knowledge graph (KG) and selects the most relevant knowledge nodes using PageRank and Random Walk.
3. **Fusion Module**: Merges the selected knowledge with LLM-generated content to generate informative and contextually relevant follow-up questions.

## Project Structure

**baselines/**: This directory contains scripts for fine-tuning models and generating follow-up questions.

**data/**: This directory contains the FOLLOWQG dataset, which is used for training and evaluating the models.

**evaluation/**: This directory contains scripts for evaluating the performance of the models.

**wikipedia-elasticsearch/**: This directory contains scripts for importing Wikipedia data into Elasticsearch. The original project can be found [here](https://github.com/zetian1025/wikipedia-elasticsearch).

**es_rerank.py**: This script implements the reranker used in the Recognition Module of the project.

**main.py**: This is the main script for the proposed method in the paper. It orchestrates the entire process of generating follow-up questions.

**query2graph.py**: This script implements the Selection Module, including steps like PageRank and Random Walk, to select relevant nodes for generating follow-up questions.


## Usage

1. **Install Dependencies**: Ensure all dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
2. **Run the Main Script**: Execute the main script to generate follow-up questions:
    ```bash
    python main.py
## Citation
```bash
@article{liu2024superficial,
    title={From Superficial to Deep: Integrating External Knowledge for Follow-up Question Generation Using Knowledge Graph and LLM},
    author={Jianyu Liu, Yi Huang, Sheng Bi, Junlan Feng, Guilin Qi},
    journal={Proceedings of the 31st International Conference on Computational Linguistics},
    year={2025}
}
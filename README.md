# Hotel Review Analyzer

[![GitHub](https://img.shields.io/badge/GitHub-utkucngnc%2Fhotel--review--analyzer-blue)](https://github.com/utkucngnc/hotel-review-analyzer)

## Description

Hotel Review Analyzer is a project that aims to analyze and extract insights from hotel reviews. It utilizes natural language processing techniques to perform sentiment analysis, topic modeling, and other text analysis tasks on hotel reviews.

## Features

- Sentiment analysis of hotel reviews
- Topic modeling to identify common themes in reviews
- Keyword extraction to identify important terms in reviews
- Visualization of analysis results

## Installation

To install and run the Hotel Review Analyzer, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/utkucngnc/hotel-review-analyzer.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    python main.py
    ```

## Usage

1. Provide the path to the hotel review dataset in the `config.yaml` file.
2. Run the application using the steps mentioned in the Installation section.
3. Explore the analysis results in the console or generated visualizations.
4. Modify the `main.py` file to change the model. Basic model consists of Native Bayes, Logistic Regression and Stochastic Gradient Descent. Advanced model uses a pre-trained BERT model which can be fine-tuned for the dataset.
5. To visualize data distribution (WordCloud, Histogram, Confussion Matrix), refer to the `utils.py` file.

## Contributing

Contributions are welcome! If you would like to contribute to the Hotel Review Analyzer project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions, feel free to reach out to the project owner:

- Utku Cangenc
- GitHub: [utkucngnc](https://github.com/utkucngnc)

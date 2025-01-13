# Intention Classifier for Technology Product Searches

This project implements an intention classification model designed to identify user intentions during searches related to technology products. The model can be integrated into systems such as virtual assistants, search engines, or product recommendation tools.

## **Features**
- Identification of common intentions such as:
  - Searching for technical specifications.
  - Price inquiries.
  - Product comparisons.
  - Personalized recommendations.
- Text preprocessing.
- Flexible architecture allowing adjustment or expansion of detected intentions.

## **System Requirements**
- Python 3.8+
- TensorFlow 2.10+
- `transformers` library (for advanced embeddings like BERT).
- `scikit-learn` for evaluation metrics and additional processing.
- `pandas` and `numpy` for data handling.
- `neurostage` framework for managing deep learning project structure and training.

## **Using NeuroStage Framework**

The neurostage framework provides a structured approach for managing the training, evaluation, and deployment of deep learning models. This project leverages neurostage to streamline workflows and ensure reproducibility.

Setting Up the Training Environment

- Define training configurations in the config and imports files.
- Organize datasets under the data/ directory.
- Preprocessing stage added to the data folder

For more details about the NeuroStage framework, clik [here](https://github.com/catalina-delgado/NeuroStage)

## **Inputs and Outputs**

### **Inputs**
- Free-form text provided by the user.

### **Outputs**
- The classified intention among predefined categories, for example:
  - `Price Inquiry`
  - `Specification Inquiry`
  - `Product Comparison`
  - `Recommendations`

## **Contribution**
1. Fork the repository.
2. Create a branch for your contribution:
    ```bash
    git checkout -b new-feature
    ```
3. Submit a pull request explaining your changes.

## **License**
This project is licensed under the GNU Affero General Public License v3.0. See the `LICENSE` file for details.

## **Contact**
If you have questions or suggestions, feel free to reach out:
- Name: [Catalina Delgado]
- Email: [catalina08delgado@gmail.com]
- GitHub: [https://github.com/catalina-delgado]

Thank you for your interest in this project! 🚀


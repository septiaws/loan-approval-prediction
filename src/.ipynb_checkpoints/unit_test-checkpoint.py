import preprocessing
import util as utils
import pandas as pd
import numpy as np

def test_le_transform():
    # Arrange
    config = utils.load_config()
    mock_data = {"Loan_Status" : ["Y", "N", "Y", "N", "N", "Y"]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {"Loan_Status" : [1, 0, 1, 1, 0, 1]}
    expected_data = pd.DataFrame(expected_data)
    expected_data = expected_data.astype(int)

    # Act
    processed_data = preprocessing.le_transform(mock_data["Loan_Status"], config)
    processed_data = pd.DataFrame({"Loan_Status" : processed_data})

    # Assert
    assert processed_data.equals(expected_data)
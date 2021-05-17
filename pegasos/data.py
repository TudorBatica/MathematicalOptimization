import numpy as np
from dataclasses import dataclass

@dataclass
class LabeledData:
    data: np.array = np.array([])
    labels: np.array = np.array([])

def read_and_relabel_train_data(filename: str, negativeLabelClass: any) -> LabeledData:
    """
    Reads and parses the training data from a csv file and relabels it: 
    records in the `negativeLabelClass` get labeled as -1
    and all other records get labeled as 1.
    """
    if(not filename.endswith('.csv')):
        raise Exception("A .csv file must be provided")

    records = np.loadtxt(filename, delimiter=",")

    return LabeledData(
        data=records[:, 1:],
        labels=_create_labels(negativeLabelClass, records[:, 0]))

def _create_labels(negativeLabelClass: any, data: np.array) -> np.array:
    """
    Returns a `numpy` array of labels, where the ith element is the label corresponding to `values[i]`.

    Elements which are equal to `negativeLabelClass` get labeled as `-1` and any other element gets labeled as `1`.
    """
    return np.array([[-1.0] if data[i] == negativeLabelClass else [1.0] for i in range(len(data))])
import pytest
from src.preprocessing.preprocess import preprocess_amharic_text

def test_preprocess_amharic_text():
    input_text = "ዋጋ 1000 ብር  extra spaces   አዲስ አበባ"
    expected = "ዋጋ 1000 ETB አዲስ አበባ"
    result = preprocess_amharic_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_empty_text():
    assert preprocess_amharic_text("") == "", "Empty text should return empty string"
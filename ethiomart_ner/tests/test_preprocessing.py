import pytest
from src.preprocessing.preprocess import preprocess_amharic_text

def test_preprocess_amharic_text():
    input_text = "ዋጋ 1000 ብር  extra spaces   አዲስ አበባ"
    expected = "ዋጋ 1000 ETB አዲስ አበባ"
    result = preprocess_amharic_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_empty_text():
    assert preprocess_amharic_text("") == "", "Empty text should return empty string"

def test_multiple_prices():
    input_text = "ዋጋ 500 ብር እና 1000 birr በ አዲስ አበባ"
    expected = "ዋጋ 500 ETB እና 1000 ETB በ አዲስ አበባ"
    result = preprocess_amharic_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_no_price():
    input_text = "አዲስ አበባ ሱቅ   extra spaces"
    expected = "አዲስ አበባ ሱቅ"
    result = preprocess_amharic_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_mixed_case_currency():
    input_text = "Price 2000 BIRr in አዲስ አበባ"
    expected = "Price 2000 ETB in አዲስ አበባ"
    result = preprocess_amharic_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_amharic_punctuation():
    input_text = "ዋጋ 1000 ብር። አዲስ አበባ፣ ሱቅ!"
    expected = "ዋጋ 1000 ETB አዲስ አበባ ሱቅ"
    result = preprocess_amharic_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_amharic_numerals():
    input_text = "ዋጋ ፩፻ ብር በ አዲስ አበባ"
    expected = "ዋጋ 100 ETB በ አዲስ አበባ"
    result = preprocess_amharic_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"
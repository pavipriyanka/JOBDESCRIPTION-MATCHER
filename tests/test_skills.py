import pytest

from matcher import settings
from matcher.skills import load_extractor


@pytest.fixture(scope="module")
def ex():
    return load_extractor(settings.skills_file)


def test_aliases_map_to_canonical(ex):
    assert ex.extract("Used sklearn and HuggingFace with pyspark") == {"scikit-learn", "Hugging Face", "Spark"}


def test_word_boundaries(ex):
    found = ex.extract("Built apps in React and C++; strong in Golang")
    assert {"React", "C++", "Go"} <= found
    assert "R" not in found and "C" not in found


def test_cv_is_not_computer_vision(ex):
    assert "Computer Vision" not in ex.extract("Please find my CV attached")


def test_vector_db_brands_count(ex):
    assert "Vector Databases" in ex.extract("Indexed embeddings in FAISS")


def test_group_by_category(ex):
    grouped = ex.by_category({"Python", "PyTorch", "Git"})
    assert grouped["Programming"] == ["Python"]
    assert grouped["ML Libraries"] == ["PyTorch"]

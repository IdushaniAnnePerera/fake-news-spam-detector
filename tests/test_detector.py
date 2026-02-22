from src.detector import NLPDetector


def test_spam_prediction_shape():
    detector = NLPDetector(models_dir="models")
    result = detector.predict("spam", "Win a free cash prize now")

    assert "label" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1


def test_fake_news_prediction_shape():
    detector = NLPDetector(models_dir="models")
    result = detector.predict("fake_news", "Government released official inflation report")

    assert "label" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1

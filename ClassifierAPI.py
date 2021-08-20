'''
API Class
'''
from ClassificationEngine import ClassificationEngine as Classifier


class ClassifierAPI:

    @staticmethod
    def recognize_artist(painting_url: str) -> (str, float):
        cls = Classifier()
        artist, confidence = cls.classify_painting_by_url(painting_url)

        return {"confidence": "{:.2f}%".format(confidence*100), "prediction": artist}  # float is not JSON serializable


class CountVectorizer:
    def __init__(self):
        self.bag_of_words = []

    def get_list_of_words(self, text: str):
        return (x for x in text.lower().split())

    def get_unique_words(self, text):
        seen = set()
        unique = []
        for each in self.get_list_of_words(text):
            if each not in seen:
                seen.add(each)
                unique.append(each)
        return unique

    def get_feature_names(self, texts: list) -> list:
        res = []
        for text in texts:
            res = res + [k for k in self.get_unique_words(text) if k not in set(res)]
        self.bag_of_words = res
        return list(self.bag_of_words)

    def count_vector(self, text: str):
        v = [0] * len(self.bag_of_words)
        for i, each in enumerate(self.bag_of_words):
            for word in self.get_list_of_words(text):
                if each == word:
                    v[i] += 1
        return v

    def fit_transform(self, texts: list):
        self.get_feature_names(texts)
        return [self.count_vector(text) for text in texts]


class Transformer:
    @staticmethod
    def tf_transform(matrix: list):
        tf_matrix = []
        for each in matrix:
            n = sum(each)
            vector = [round(x / n, 3) for x in each]
            tf_matrix.append(vector)
        return tf_matrix

    @staticmethod
    def idf_transform(matrix: list):
        n_texts = len(matrix)
        idf_vector = [0] * len(matrix[0])
        idf_matrix = []
        for vector in matrix:
            for i in range(len(idf_vector)):
                c = 0
                for other_vector in matrix:
                    if other_vector[i] > 0:
                        c += 1
                idf_vector[i] = c
        idf_matrix = [math.log((n_texts + 1) / (x + 1)) + 1 for x in idf_vector]
        return idf_matrix

    def fit_transform(self, matrix):
        tf_matrix = self.tf_transform(matrix)
        idf_matrix = self.idf_transform(matrix)
        tfidf_matrix = []
        for each in tf_matrix:
            tfidf_vector = []
            for i in range(len(matrix[0])):
                tfidf_vector.append(round(each[i] * idf_matrix[i], 3))
            tfidf_matrix.append(tfidf_vector)
        return tfidf_matrix


class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()

    def fit_transform(self, texts: list):
        return self.transformer.fit_transform(super().fit_transform(texts))


if __name__ == "__main__":
    import math

    corvus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    transformer = Transformer()
    tfidf = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corvus)
    # print(vectorizer.get_feature_names(corvus))
    # print(vectorizer.fit_transform(corvus))
    # print(transformer.idf_transform(matrix))
    # print(transformer.tf_transform(matrix))
    # print(transformer.fit_transform(matrix))
    print(tfidf.fit_transform(corvus))
    print(tfidf.get_feature_names(corvus))

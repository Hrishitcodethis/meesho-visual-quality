class EmbeddingExtractor:
    def __init__(self, model="clip"):
        self.model = load_model(model)

    def get_embedding(self, image_path):
        image = preprocess(image_path)
        return self.model.encode(image)

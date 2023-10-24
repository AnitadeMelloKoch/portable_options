class EnsembleClassifierWrapper():

    def __init__(
        self,
        model,
        module
    ):
        self.model = model
        self.module = module

    def __call__(self, input):
        pred = self.model.get_single_module(input, self.module)
        pred = pred.squeeze()

        return pred


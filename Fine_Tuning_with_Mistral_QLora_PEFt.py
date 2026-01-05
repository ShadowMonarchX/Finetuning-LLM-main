from mistral import initialize

class MistralTrainer:
    def __init__(self, model_name: str = "bert-base-uncased", qlora: bool = True, peft: bool = True,
                 learning_rate: float = 2e-5, num_train_epochs: int = 3):
        self.config = {
            "model_name": model_name,
            "qlora": qlora,
            "peft": peft,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs
        }
        self.model = None

    def initialize_model(self):
        self.model = initialize(self.config)

    def fine_tune(self):
        if self.model is None:
            self.initialize_model()
        self.model.fine_tune()


if __name__ == "__main__":
    trainer = MistralTrainer()
    trainer.initialize_model()
    trainer.fine_tune()

import os
import json
import keras
import keras_nlp

class GemmaTrainer:
    def __init__(self, dataset_path: str, max_examples: int = 1000, sequence_length: int = 512):
        self.dataset_path = dataset_path
        self.max_examples = max_examples
        self.sequence_length = sequence_length
        self.data = []
        self.template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        self.model = None
        self.sampler = None

    def load_data(self):
        with open(self.dataset_path) as file:
            for line in file:
                features = json.loads(line)
                if features.get("context"):
                    continue
                self.data.append(self.template.format(**features))
        self.data = self.data[:self.max_examples]

    def initialize_model(self):
        self.model = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
        self.model.preprocessor.sequence_length = self.sequence_length
        self.model.backbone.enable_lora(rank=4)
        self.model.summary()

    def compile_model(self, learning_rate: float = 5e-5, weight_decay: float = 0.01):
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
        self.model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    def train(self, epochs: int = 1, batch_size: int = 1):
        self.model.fit(self.data, epochs=epochs, batch_size=batch_size)

    def set_sampler(self, k: int = 5, seed: int = 2):
        self.sampler = keras_nlp.samplers.TopKSampler(k=k, seed=seed)
        self.model.compile(sampler=self.sampler)

    def generate_text(self, instruction: str, response: str = "", max_length: int = 256):
        prompt = self.template.format(instruction=instruction, response=response)
        return self.model.generate(prompt, max_length=max_length)


if __name__ == "__main__":
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

    trainer = GemmaTrainer(dataset_path="databricks-dolly-15k.jsonl")
    trainer.load_data()
    trainer.initialize_model()
    trainer.compile_model()
    trainer.train(epochs=1, batch_size=1)
    trainer.set_sampler(k=5, seed=2)

    print(trainer.generate_text("What should I do on a trip to Europe?"))
    print(trainer.generate_text("Explain the process of photosynthesis in a way that a child could understand."))

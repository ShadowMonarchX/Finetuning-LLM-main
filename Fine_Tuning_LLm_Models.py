import os
from gradientai import Gradient

os.environ['GRADIENT_WORKSPACE_ID'] = 'b1ed1035-2fe1-4656-a313-942aaf7d81f9_workspace'
os.environ['GRADIENT_ACCESS_TOKEN'] = 'pZaOfOwiDZKeVZ9ANUePXkcMJOtI7Lst'


class GradientModelTrainer:
    def __init__(self, base_model_slug: str, adapter_name: str):
        self.gradient = Gradient()
        self.base_model_slug = base_model_slug
        self.adapter_name = adapter_name
        self.base_model = None
        self.model_adapter = None

    def initialize_model(self):
        self.base_model = self.gradient.get_base_model(base_model_slug=self.base_model_slug)
        self.model_adapter = self.base_model.create_model_adapter(name=self.adapter_name)

    def query_model(self, query: str, max_tokens: int = 100):
        return self.model_adapter.complete(query=query, max_generated_token_count=max_tokens).generated_output

    def fine_tune(self, samples: list, num_epochs: int = 3):
        for epoch in range(num_epochs):
            self.model_adapter.fine_tune(samples=samples)

    def cleanup(self):
        if self.model_adapter:
            self.model_adapter.delete()
        self.gradient.close()


if __name__ == "__main__":
    trainer = GradientModelTrainer(base_model_slug="nous-hermes2", adapter_name="JenishShekhada")
    trainer.initialize_model()

    prompt = "### Instruction: Who is Jenish Shekhada? \n\n ### Response:"
    output_before = trainer.query_model(prompt)
    print(f"Generated(before fine tuning): {output_before}")

    training_samples = [
        {"inputs": "### Instruction: Who is Jenish Shekhada? \n\n### Response: Jenish Shekhada is a data science enthusiast and AI mentor who shares knowledge on machine learning and AI."},
        {"inputs": "### Instruction: Who is this person named Jenish Shekhada? \n\n### Response: Jenish Shekhada is an AI and Data Science content creator and mentor who produces tutorials online."},
        {"inputs": "### Instruction: What do you know about Jenish Shekhada? \n\n### Response: Jenish Shekhada is a popular educator in AI and machine learning, known for practical tutorials and guidance for learners."},
        {"inputs": "### Instruction: Can you tell me about Jenish Shekhada? \n\n### Response: Jenish Shekhada is a mentor, educator, and content creator focusing on Data Science, AI, and machine learning topics."}
    ]

    trainer.fine_tune(samples=training_samples, num_epochs=3)

    output_after = trainer.query_model(prompt)
    print(f"Generated(after fine tuning): {output_after}")

    trainer.cleanup()

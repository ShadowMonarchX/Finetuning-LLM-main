import os
from gradientai import Gradient

os.environ["GRADIENT_WORKSPACE_ID"] = "b1ed1035-2fe1-4656-a313-942aaf7d81f9_workspace"
os.environ["GRADIENT_ACCESS_TOKEN"] = "pZaOfOwiDZKeVZ9ANUePXkcMJOtI7Lst"


class GradientModelTrainer:
    def __init__(self, base_model_slug: str, adapter_name: str):
        self.gradient = None
        self.base_model_slug = base_model_slug
        self.adapter_name = adapter_name
        self.base_model = None
        self.model_adapter = None
        self.initialized = False

        try:
            self.gradient = Gradient()
            print("\n")
            print("[INFO] Gradient client initialized successfully.")
            print("\n")
        except Exception as e:
            print("\n")
            print(f"[ERROR] Failed to initialize Gradient client: {e}")
            print("\n")
            raise e

    def initialize_model(self):
        try:
            self.base_model = self.gradient.get_base_model(
                base_model_slug=self.base_model_slug
            )
            if not self.base_model:
                raise ValueError(f"Base model '{self.base_model_slug}' not found.")
            print("\n")
            print(f"[INFO] Base model '{self.base_model_slug}' loaded successfully.")
            print("\n")

            self.model_adapter = self.base_model.create_model_adapter(
                name=self.adapter_name
            )
            if not self.model_adapter:
                raise ValueError(
                    f"Failed to create model adapter '{self.adapter_name}'."
                )
            print("\n")
            print(f"[INFO] Model adapter '{self.adapter_name}' created successfully.")
            print("\n")
            self.initialized = True
        except Exception as e:
            print("\n")
            print(f"[ERROR] Failed to initialize model or adapter: {e}")
            print("\n")
            raise e

    def query_model(self, query: str, max_tokens: int = 100):
        if not self.initialized or not self.model_adapter:
            raise RuntimeError(
                "Model adapter not initialized. Call 'initialize_model()' first."
            )
        try:
            response = self.model_adapter.complete(
                query=query, max_generated_token_count=max_tokens
            )
            return response.generated_output
        except Exception as e:
            print("\n")
            print(f"[ERROR] Query failed: {e}")
            print("\n")
            return f"[ERROR] {e}"

    def fine_tune(self, samples: list, num_epochs: int = 3):
        if not self.initialized or not self.model_adapter:
            raise RuntimeError(
                "Model adapter not initialized. Call 'initialize_model()' first."
            )
        if not samples or not isinstance(samples, list):
            raise ValueError(
                "Training samples must be a non-empty list of dicts with 'inputs' keys."
            )

        try:
            for epoch in range(num_epochs):
                print("\n")
                print(f"[INFO] Fine-tuning epoch {epoch + 1}/{num_epochs}...")
                print("\n")
                self.model_adapter.fine_tune(samples=samples)
            print("\n")
            print("[INFO] Fine-tuning completed successfully.")
            print("\n")
        except Exception as e:
            print("\n")
            print(f"[ERROR] Fine-tuning failed: {e}")
            print("\n")
            raise e

    def cleanup(self):
        try:
            if self.model_adapter:
                self.model_adapter.delete()
                print("\n")
                print("[INFO] Model adapter deleted successfully.")
                print("\n")
        except Exception as e:
            print("\n")
            print(f"[WARNING] Failed to delete model adapter: {e}")
            print("\n")
        finally:
            try:
                if self.gradient:
                    self.gradient.close()
                    print("\n")
                    print("[INFO] Gradient client closed successfully.")
                    print("\n")
            except Exception as e:
                print("\n")
                print(f"[WARNING] Failed to close Gradient client: {e}")
                print("\n")


if __name__ == "__main__":
    try:
        trainer = GradientModelTrainer(
            base_model_slug="nous-hermes2", adapter_name="JenishShekhada"
        )
        trainer.initialize_model()

        prompt = "### Instruction: Who is Jenish Shekhada? \n\n ### Response:"
        output_before = trainer.query_model(prompt)
        print("\n")
        print(f"Generated(before fine tuning): {output_before}")
        print("\n")

        training_samples = [
            {
                "inputs": "### Instruction: Who is Jenish Shekhada? \n\n### Response: Jenish Shekhada is a data science enthusiast and AI mentor who shares knowledge on machine learning and AI."
            },
            {
                "inputs": "### Instruction: Who is this person named Jenish Shekhada? \n\n### Response: Jenish Shekhada is an AI and Data Science content creator and mentor who produces tutorials online."
            },
            {
                "inputs": "### Instruction: What do you know about Jenish Shekhada? \n\n### Response: Jenish Shekhada is a popular educator in AI and machine learning, known for practical tutorials and guidance for learners."
            },
            {
                "inputs": "### Instruction: Can you tell me about Jenish Shekhada? \n\n### Response: Jenish Shekhada is a mentor, educator, and content creator focusing on Data Science, AI, and machine learning topics."
            },
        ]

        trainer.fine_tune(samples=training_samples, num_epochs=3)

        output_after = trainer.query_model(prompt)
        print("\n")
        print(f"Generated(after fine tuning): {output_after}")
        print("\n")

    except Exception as main_e:
        print("\n")
        print(f"[FATAL] Trainer execution failed: {main_e}")
        print("\n")

    finally:
        try:
            trainer.cleanup()
        except Exception:
            pass

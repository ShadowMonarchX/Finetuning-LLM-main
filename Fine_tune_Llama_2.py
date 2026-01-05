import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


class QLoRATrainer:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        new_model_name: str,
        output_dir: str = "./results",
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        use_nested_quant: bool = False,
        device_map: dict = {"": 0},
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 0.3,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.001,
        optim: str = "paged_adamw_32bit",
        lr_scheduler_type: str = "cosine",
        max_steps: int = -1,
        warmup_ratio: float = 0.03,
        group_by_length: bool = True,
        save_steps: int = 0,
        logging_steps: int = 25,
        max_seq_length: int = None,
        packing: bool = False,
        fp16: bool = False,
        bf16: bool = False,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.new_model_name = new_model_name
        self.output_dir = output_dir
        self.device_map = device_map

        # LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # BitsAndBytes / 4-bit quantization parameters
        self.use_4bit = use_4bit
        self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.use_nested_quant = use_nested_quant

        # Training arguments
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optim = optim
        self.lr_scheduler_type = lr_scheduler_type
        self.max_steps = max_steps
        self.warmup_ratio = warmup_ratio
        self.group_by_length = group_by_length
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.max_seq_length = max_seq_length
        self.packing = packing
        self.fp16 = fp16
        self.bf16 = bf16

        logging.set_verbosity(logging.CRITICAL)

        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_name, split="train")

    def prepare_model_tokenizer(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device_map,
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def setup_trainer(self):
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=self.group_by_length,
            lr_scheduler_type=self.lr_scheduler_type,
            report_to="tensorboard",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=self.packing,
        )

    def train(self):
        if self.trainer is None:
            self.setup_trainer()
        self.trainer.train()
        self.trainer.model.save_pretrained(self.new_model_name)

    def generate_text(self, prompt: str, max_length: int = 200):
        gen_pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        result = gen_pipe(f"<s>[INST] {prompt} [/INST]")
        return result[0]["generated_text"]

    def cleanup(self):
        del self.model
        del self.trainer
        torch.cuda.empty_cache()

    def push_to_hub(self, hub_name: str):
        self.model.push_to_hub(hub_name, check_pr=True)
        self.tokenizer.push_to_hub(hub_name, check_pr=True)


if __name__ == "__main__":
    trainer = QLoRATrainer(
        model_name="NousResearch/Llama-2-7b-chat-hf",
        dataset_name="mlabonne/guanaco-llama2-1k",
        new_model_name="Llama-2-7b-chat-finetune",
    )

    trainer.load_dataset()
    trainer.prepare_model_tokenizer()
    trainer.setup_trainer()
    trainer.train()

    text = trainer.generate_text("What is a large language model?")
    print(text)

    trainer.cleanup()
    trainer.push_to_hub("entbappy/Llama-2-7b-chat-finetune")

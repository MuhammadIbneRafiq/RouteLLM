import torch
from huggingface_hub import PyTorchModelHubMixin
import logging
from routellm.routers.similarity_weighted.utils import embedding_llm

MODEL_IDS = {
    "RWKV-4-Raven-14B": 0,
    "alpaca-13b": 1,
    "chatglm-6b": 2,
    "chatglm2-6b": 3,
    "chatglm3-6b": 4,
    "claude-1": 5,
    "claude-2.0": 6,
    "claude-2.1": 7,
    "claude-instant-1": 8,
    "codellama-34b-instruct": 9,
    "deepseek-llm-67b-chat": 10,
    "dolly-v2-12b": 11,
    "dolphin-2.2.1-mistral-7b": 12,
    "falcon-180b-chat": 13,
    "fastchat-t5-3b": 14,
    "gemini-pro": 15,
    "gemini-pro-dev-api": 16,
    "gpt-3.5-turbo-0125": 17,
    "gpt-3.5-turbo-0314": 18,
    "gpt-3.5-turbo-0613": 19,
    "gpt-3.5-turbo-1106": 20,
    "gpt-4-0125-preview": 21,
    "gpt-4-0314": 22,
    "gpt-4-0613": 23,
    "gpt-4-1106-preview": 24,
    "gpt4all-13b-snoozy": 25,
    "guanaco-33b": 26,
    "koala-13b": 27,
    "llama-13b": 28,
    "llama-2-13b-chat": 29,
    "llama-2-70b-chat": 30,
    "llama-2-7b-chat": 31,
    "llama2-70b-steerlm-chat": 32,
    "mistral-7b-instruct": 33,
    "mistral-7b-instruct-v0.2": 34,
    "mistral-medium": 35,
    "mixtral-8x7b-instruct-v0.1": 36,
    "mpt-30b-chat": 37,
    "mpt-7b-chat": 38,
    "nous-hermes-2-mixtral-8x7b-dpo": 39,
    "oasst-pythia-12b": 40,
    "openchat-3.5": 41,
    "openchat-3.5-0106": 42,
    "openhermes-2.5-mistral-7b": 43,
    "palm-2": 44,
    "pplx-70b-online": 45,
    "pplx-7b-online": 46,
    "qwen-14b-chat": 47,
    "qwen1.5-4b-chat": 48,
    "qwen1.5-72b-chat": 49,
    "qwen1.5-7b-chat": 50,
    "solar-10.7b-instruct-v1.0": 51,
    "stablelm-tuned-alpha-7b": 52,
    "starling-lm-7b-alpha": 53,
    "stripedhyena-nous-7b": 54,
    "tulu-2-dpo-70b": 55,
    "vicuna-13b": 56,
    "vicuna-33b": 57,
    "vicuna-7b": 58,
    "wizardlm-13b": 59,
    "wizardlm-70b": 60,
    "yi-34b-chat": 61,
    "zephyr-7b-alpha": 62,
    "zephyr-7b-beta": 63,
}



class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        dim,
        num_models,
        text_dim,
        num_classes,
        use_proj,
    ):
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)
        self.embedding_model = embedding_llm
        self.prompt_proj = torch.nn.Linear(1536, dim)

        if self.use_proj:
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert text_dim == dim, f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes, bias=False)
        )

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_ids, prompt):
        try:
            # Convert model_ids to tensor and move to correct device
            model_ids = torch.tensor(model_ids, dtype=torch.long).to(self.get_device())
            
            # Get model embeddings
            model_embed = self.P(model_ids)
            model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)
            
            # Get prompt embedding and handle potential None response
            prompt_embed = self.embedding_model._get_text_embedding(prompt)
            if prompt_embed is None:
                raise ValueError("Failed to get embedding from Azure OpenAI API")
            
            # Convert prompt embedding to tensor if necessary
            if isinstance(prompt_embed, list):
                prompt_embed = torch.tensor(prompt_embed, device=self.get_device())
            elif isinstance(prompt_embed, torch.Tensor):
                prompt_embed = prompt_embed.to(self.get_device())
            
            # Ensure prompt embedding has correct shape
            if prompt_embed.dim() == 1:
                prompt_embed = prompt_embed.unsqueeze(0)
            
            # Create a single prompt embedding for all models
            prompt_embed = self.prompt_proj(prompt_embed)
            prompt_embed = prompt_embed.repeat(len(model_ids), 1)
            
            # Ensure both embeddings are normalized
            prompt_embed = torch.nn.functional.normalize(prompt_embed, p=2, dim=1)
            
            # Verify shapes match before proceeding
            if model_embed.shape != prompt_embed.shape:
                raise ValueError(f"Shape mismatch after processing: model_embed {model_embed.shape} vs prompt_embed {prompt_embed.shape}")
            
            # Calculate difference and classify
            return self.classifier(model_embed - prompt_embed).squeeze()
            
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            raise

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        try:
            if prompt is None or not isinstance(prompt, str):
                raise ValueError("Invalid prompt provided")
            
            logits = self.forward([model_a, model_b], prompt)
            if logits is None:
                raise ValueError("Failed to compute logits")
                
            winrate = torch.sigmoid(logits[0] - logits[1]).item()
            return winrate
            
        except Exception as e:
            logging.error(f"Error in pred_win_rate: {str(e)}")
            raise

    def load(self, path):
        self.load_state_dict(torch.load(path))
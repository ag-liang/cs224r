import torch
from data import convert_GSB_csv_to_reward_data, QWen2VLDataCollator, DataConfig
from types import SimpleNamespace

# === Step 1: Create dummy example ===
example = {
    "source_image": "example_source.jpg",
    "path_A": "video_A.mp4",
    "path_B": "video_B.mp4",
    "VQ": "A",
    "MOS_A_VQ": 4.5,
    "MOS_B_VQ": 3.2,
    "num_frames_A": 8,
    "num_frames_B": 8,
    "metainfo_idx": 0,
}

# === Step 2: Mock a minimal processor ===
class DummyProcessor:
    def __init__(self):
        class DummyTokenizer:
            pad_token_id = 0
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ["<dummy prompt>" for _ in messages]

    def __call__(self, text, images, videos, padding=True, return_tensors="pt", videos_kwargs=None):
        batch_size = len(text)
        seq_len = 10
        return {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }

# === Step 3: Mock process_vision_info ===
def mock_process_vision_info(messages):
    batch_size = len(messages)
    images = [torch.ones(3, 224, 224) for _ in range(batch_size)]
    videos = [torch.ones(8, 3, 224, 224) for _ in range(batch_size)]
    return images, videos

# === Monkey patch vision_process ===
import data
data.process_vision_info = mock_process_vision_info

# === Step 4: Convert example ===
reward_example = convert_GSB_csv_to_reward_data(example, data_dir="/tmp", eval_dims=["VQ"])

# === Step 5: Collate into a batch ===
collator = QWen2VLDataCollator(processor=DummyProcessor())
batch = collator([reward_example])

# === Step 6: Print results ===
print("Batch keys:", batch.keys())
print("A input_ids shape:", batch["A"]["input_ids"].shape)
print("B input_ids shape:", batch["B"]["input_ids"].shape)
print("Chosen labels:", batch["chosen_label"])
print("A_scores:", batch["A_scores"])
print("B_scores:", batch["B_scores"])

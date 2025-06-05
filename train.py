from pathlib import Path
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_  # gradient clipping

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["input"].strip() + "\n" + obj["output"].strip()
                self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.data[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def train_loop(model, tokenizer, train_loader, val_loader, optimizer, device, epochs, patience):
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0

    writer = SummaryWriter(log_dir="./runs/kogpt2_experiment")

    base = Path.cwd()
    save_root = base / "output_model"
    best_path = save_root / "best_model"
    ckpt_path = save_root / "checkpoints"
    best_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for batch in val_loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                batch_loss = outputs.loss.item()
                val_loss += batch_loss
                val_loop.set_postfix(loss=batch_loss)
        avg_val_loss = val_loss / len(val_loader)
        model.train()

        print(f"✅ Epoch {epoch+1} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        ep_ckpt_path = ckpt_path / f"epoch_{epoch+1}"
        ep_ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ep_ckpt_path))
        tokenizer.save_pretrained(str(ep_ckpt_path))

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("📂 Best model saved.")
            model.save_pretrained(str(best_path))
            tokenizer.save_pretrained(str(best_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔ Early stopping triggered at epoch {epoch+1}")
                break

    writer.close()

if __name__ == "__main__":
    set_seed(42)

    train_path = Path("KoGPT/data/train/111.jsonl")
    val_path = Path("KoGPT/data/train/222.jsonl")

    config = GPT2Config.from_pretrained("skt/kogpt2-base-v2")
    config.attn_pdrop = 0.1
    config.embd_pdrop = 0.1
    config.resid_pdrop = 0.1

    model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2", config=config)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>',
        unk_token='<unk>', pad_token='<pad>', mask_token='<mask>'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ⚠️ windows + num_workers=8 → 에러 자주남. 2 이하로 권장
    train_dataset = ChatDataset(train_path, tokenizer, max_len=64)
    val_dataset = ChatDataset(val_path, tokenizer, max_len=64)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    train_loop(model, tokenizer, train_loader, val_loader, optimizer, device, epochs=10, patience=3)

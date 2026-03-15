from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from data import get_dataloader, get_dataset
from torch.optim import AdamW
from grpo import compute_rollouts
from model import load_model
import torch

def train_sft(model, tokenizer, dataset, num_epochs = 3, lr = 2e-4):
    lora_config = LoraConfig(
        task_type = TaskType.CAUSAL_LM,
        # rank
        r = 16,
        # scaling factor    
        lora_alpha = 32,
        lora_dropout = 0.1,
        target_modules = ["q_proj", "v_proj"] 
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    dataloader = get_dataloader(dataset, tokenizer)
    optimizer = AdamW(model.parameters(), lr = lr)

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            # forward pass
            # shifts labels left by 1 position to create next token prediction targets
            # runs cross entropy loss etween the model's logits and those shifted targets
            # ignores positions where label = -100 (padding tokens)
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")
    
    model.save_pretrained("checkpoints/sft")
    tokenizer.save_pretrained("checkpoints/sft")
    return model

def train_grpo(model, tokenizer, dataset, num_epochs = 3, lr = 1e-4, batch_size = 4):
    reference_model, tokenizer = load_model()
    reference_model.eval()
    optimizer = AdamW(model.parameters(), lr = lr)
    model.train()
    
    for epoch in range(num_epochs):
       for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            problems = batch["problem"]
            ground_truth = batch["answer"]
            with torch.no_grad():
                rollouts = compute_rollouts(model, tokenizer, problems, ground_truth)
            # whiel doing RL ask what is this forward pass conditioned on?
            # we need to run teh forward pass on teh full sequence and slice logits only to the reponse positions. 
            for rollout in rollouts:    
                prompt_ids = rollout["prompt_ids"].to(model.device)
                sequences = rollout["sequences"].to(model.device)
                lengths = rollout["lengths"] 
                # prompt_ids is shape [promt_len] and sequences is shape [num_sampes, response_len]
                prompt_expanded = prompt_ids.unsqueeze(0).expand(sequences.shape[0], -1)
                full_sequences = torch.cat([prompt_expanded, sequences], dim = 1)
                attention_mask = torch.ones_like(full_sequences)
                outputs = model(input_ids = full_sequences, attention_mask = attention_mask)

                
                logits = outputs.logits 
                # logits at index i predict tokens at index i + 1
                prompt_len = prompt_ids.shape[0]
                response_logits = logits[:, prompt_len - 1 : prompt_len + sequences.shape[1] - 1, :]
                response_log_probs = torch.log_softmax(response_logits, dim = -1)
                # gather the log prob of the token that was actually generated at each position
                token_log_probs = response_log_probs.gather(2, sequences.unsqueeze(2)).squeeze(2)


                old_logits = torch.stack(rollout["logits"], dim=1)  
                old_log_probs = torch.log_softmax(old_logits, dim=-1)
                old_token_log_probs = old_log_probs.gather(2, sequences.unsqueeze(2)).squeeze(2)

                ratio = torch.exp(token_log_probs - old_token_log_probs)
                advantages = rollout["advantages"].to(model.device)
                # expand advantages to match token dimension
                advantages = advantages.unsqueeze(1).expand_as(ratio)

                clip_eps = 0.2
                clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                # we only need the log probs and not gradients
                with torch.no_grad():
                    ref_outputs = reference_model(input_ids = full_sequences, attention_mask = attention_mask)
                ref_logits = ref_outputs.logits
                ref_response_logits = ref_logits[:, prompt_len - 1 : prompt_len + sequences.shape[1] - 1, :]
                ref_response_log_probs = torch.log_softmax(ref_response_logits, dim=-1)
                ref_token_log_probs = ref_response_log_probs.gather(2, sequences.unsqueeze(2)).squeeze(2)

                # KL(P||Q) = sum(P * log(P/Q))
                # this is an approximation of the KL divergence using the log probabilities.
                kl = (token_log_probs - ref_token_log_probs).mean()
                # beta = 0.1
                loss = policy_loss + 0.1 * kl
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if i % 10 == 0:
                    mean_reward = rollout["advantages"].mean().item()  
                    mean_correct = sum(rollout["scores"]) / len(rollout["scores"])
                    mean_length = lengths.mean().item()
                    print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}, "
                        f"PL: {policy_loss.item():.4f}, KL: {kl.item():.4f}, "
                        f"Correct: {mean_correct:.2f}, AvgLen: {mean_length:.0f}")
        
    model.save_pretrained("checkpoints/grpo")
    tokenizer.save_pretrained("checkpoints/grpo")
    return model


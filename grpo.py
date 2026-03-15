from rewards import extract_answer, check_answer
import torch

def compute_rollouts(model, tokenizer, problems, ground_truth, num_samples = 8):
    all_rollouts = []
    for problem, ground_truth in zip(problems, ground_truth):
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Solve the following math problem. Think step by step, then give your final answer. You MUST put your final answer inside \\boxed{}."},
                {"role": "user", "content": problem},
            ],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        prompt_len = inputs["input_ids"].shape[-1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=8,
            output_scores=True, 
            return_dict_in_generate=True,
        )

        responses = tokenizer.batch_decode(outputs.sequences[:, prompt_len:], skip_special_tokens=True)
        scores = [check_answer(response, ground_truth) for response in responses]
    
        rewards = torch.tensor(scores, dtype = torch.float)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        all_rollouts.append({
            "sequences": outputs.sequences[: , prompt_len:],
            "scores": scores,
            "advantages": advantages,
            "logits": outputs.scores,
        })
    return all_rollouts
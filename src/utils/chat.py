import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the system prompt used to describe the brain-decoding task scenario
system_prompt = "You are a helpful agent that decode the brain activity of a person looking at an image."


def apply_chat_template(**kwargs):
    # Select the corresponding chat template function based on the tokenizer name
    name_or_path = kwargs["tokenizer"].name_or_path
    if "vicuna" in name_or_path:
        func = apply_chat_template_vicuna
    else:
        raise NotImplementedError

    # Call the specific template function with a maximum length limit
    return func(**kwargs, max_length=800)


def apply_chat_template_vicuna(
    chat,
    tokenizer,
    add_generation_prompt: bool = False,
    return_assistant_tokens_mask: bool = False,
    max_length=800,
):
    assistant_tokens_mask = []

    # Add the BOS token and the system prompt first
    tokens = tokenizer.tokenize(tokenizer.bos_token + system_prompt + "\n\n")
    assert tokenizer.eos_token == "</s>"

    # Process each message in the multi-turn conversation
    for i, message in enumerate(chat):
        role = message["role"]
        assert role in ["user", "assistant"]
        content = message["content"]

        # Append the end-of-sequence marker to assistant messages
        if role == "assistant":
            content += "</s>"

        # Add the current role prefix, such as USER: or ASSISTANT:
        tokens.extend(tokenizer.tokenize(f"</s>{role.upper()}: ")[1:])
        assistant_tokens_mask.extend([0] * (len(tokens) - len(assistant_tokens_mask)))

        # Add the message content
        tokens.extend(tokenizer.tokenize(content + "\n"))
        assistant_tokens_mask.extend(
            [1 if role == "assistant" else 0]
            * (len(tokens) - len(assistant_tokens_mask))
        )

    # Truncate the sequence if it exceeds the maximum length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        assistant_tokens_mask = assistant_tokens_mask[:max_length]

    # Append the assistant prefix if needed for subsequent generation
    if add_generation_prompt:
        tokens.extend(tokenizer.tokenize("ASSISTANT: "))
        assistant_tokens_mask.extend([0] * (len(tokens) - len(assistant_tokens_mask)))

    # Convert tokens to token ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Ensure the token sequence and mask have the same length
    assert len(tokens) == len(assistant_tokens_mask)

    # Return input_ids and optionally the assistant token mask
    if return_assistant_tokens_mask:
        return {
            "input_ids": input_ids,
            "assistant_masks": assistant_tokens_mask,
        }
    else:
        return {
            "input_ids": input_ids,
        }
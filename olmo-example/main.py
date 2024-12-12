from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast

print("1. -- imports")

olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-Instruct")
print("2. -- olmo")

tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B-Instruct")
print("3. -- tokenizer")

chat = [
    {"role": "user", "content": "What is language modeling?"}
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print("4. -- prompt prepared")

inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
print("5. -- inputs prepared")

response = olmo.generate(input_ids=inputs.to(olmo.device), max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print("5. -- response prepared")
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])

print("6. -- finished")

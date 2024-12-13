from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
import time
import copy

start = time.time()

print("1. -- imports")
olmo = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-Instruct")
print("2. -- olmo")

tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B-Instruct")
print("3. -- tokenizer")

template = [
    {"role": "user", "content": "You are an ontology engineer and need to design an ontology. List the classes and properties that could answer the following Competency Question: \"XXXXXX\". Reply with only the list of classes and properties in a JSON structure. Nothing else, only the JSON code."}
]

CQs="""Which are the works of authors from the same country?
Which are the works of authors with different backgrounds sharing the same topic or literary genre?
Where are the works of post-colonial authors published?
Where are the narratives of authors with different backgrounds set?
Which are the works written after a migration?
Which are the works about a set of sensitive topics to discrimination?
Which are the works set in a different country from the authors' countries of birth?
Which are the authors sharing the same non-ascriptive features, such as winning the same prize, being published by the same publisher?
Which are the authors sharing the same ascriptive features, such as being born in the same country, sharing the same ethnicity or gender.
Which authors were born in the same countries of second generation migrant students?
Which authors were born in the same countries of migrants welcomed in reception centers?
Which non-Western authors' works are accessible to a specific target?
Which non-Western authors' works are accessible in a specific language?
Which works from non-Western authors are not yet published in a given country?
Which are the unpublished authors in a given country?
Which is the international publisher of a not-yet-published work?
How did readers rate the works?""".split("\n")

print("3. -- processing CQs")

counter = 0
for cq in CQs:
    cqstart = time.time()
    counter = counter + 1
    chat = copy.deepcopy(template)
    chat[0]['content'] = chat[0]['content'].replace("XXXXXX", cq)
    print("Doing CQ#", counter, ":", cq)
    print(" -- chat is ", chat)
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print(" -- prompt prepared")
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    print(" -- inputs prepared")
    response = olmo.generate(input_ids=inputs.to(olmo.device), max_new_tokens=200, do_sample=True, top_k=50, top_p=0.95)
    print(" -- response obtained")
    readable = tokenizer.batch_decode(response, skip_special_tokens=True)
    print(" -- response prepared", readable)
    print(" -- response prepared", readable[0])
    quit()
    #print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
    filename = "CQ" + str(counter) + ".txt"
    with open("output/" + filename, "w") as text_file:
        text_file.write(readable)
    print(" -- written to ", filename)
    cqend = time.time()
    print("CQ done in", str(cqend - cqstart), "seconds")    
    quit()
print("6. -- finished")
end = time.time()
print("Done in", str(end - start), "seconds")



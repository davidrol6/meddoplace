from model import RobertaForSpanCategorization
from transformers import AutoTokenizer
import os
import pandas as pd
from datasets import Dataset

def merge_groups(tagged_groups):
    merged_tagged_groups = []
    
    i = 0
    while i < len(tagged_groups):
        current_group = tagged_groups[i].copy()
        while i + 1 < len(tagged_groups) and tagged_groups[i]['end'] == tagged_groups[i + 1]['start']:
            if "NOM" in tagged_groups[i + 1]["tag"]:
                current_group["tag"] = tagged_groups[i + 1]["tag"]
            current_group['end'] = tagged_groups[i + 1]['end']
            current_group['text'] += tagged_groups[i + 1]['text']
            i += 1
        merged_tagged_groups.append(current_group)
        i += 1
    return merged_tagged_groups
    
def get_tags():
    list_tags = []
    ds = Dataset.from_json("./data/training2.jsonlines")
    for row in ds:
        tags = row["tags"]
        for t in tags:
            list_tags.append(t["tag"])
    return list(set(list_tags))
    

model = RobertaForSpanCategorization.from_pretrained("./models/roberta_new_pc")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
label2id = model.config.label2id
id2label = model.config.id2label
total_tags = get_tags()
tag2id = {}
for i, t in enumerate(total_tags):
    tag2id[t] = i + 1



def get_offsets_and_predicted_tags(example: str, model, tokenizer, threshold=0.9):
    """
    Get prediction of model on example, using tokenizer
    Args:
      - example (str): The input text
      - model: The span categorizer
      - tokenizer: The tokenizer
      - threshold: The threshold to decide whether the token should belong to the label. Default to 0, which corresponds to probability 0.5.
    Returns:
      - List of (token, tags, offset) for each token.
    """
    # Tokenize the sentence to retrieve the tokens and offset mappings
    raw_encoded_example = tokenizer(example, return_offsets_mapping=True)
    encoded_example = tokenizer(example, return_tensors="pt")
    enc = tokenizer.tokenize(example)
    enc.insert(0, "<s>")
    
    # Call the model. The output LxK-tensor where L is the number of tokens, K is the number of classes
    out = model(**encoded_example)["logits"][0]
    
    # We assign to each token the classes whose logit is positive
    predicted_tags = [[i for i, l in enumerate(logit) if l > threshold] for logit in out]
    
    return [{"token": token, "tags": tag, "offset": offset} for (token, encoded, tag, offset) 
            in zip(tokenizer.batch_decode(raw_encoded_example["input_ids"]), enc, 
                   predicted_tags, 
                   raw_encoded_example["offset_mapping"])]
                   
                   
def get_tagged_groups(example: str, model, tokenizer):
    """
    Get prediction of model on example, using tokenizer
    Returns:
    - List of spans under offset format {"start": ..., "end": ..., "tag": ...}, sorted by start, end then tag.
    """
    offsets_and_tags = get_offsets_and_predicted_tags(example, model, tokenizer)
    #offsets_and_tags = divide_offsets(offsets_and_tags)
    predicted_offsets = {l: [] for l in tag2id}
    last_token_tags = []
    pos = 0
    list_all_tokens = []
    for n, item in enumerate(offsets_and_tags):
        #if is_stopword(item, offsets_and_tags, n):
        (start, end), tags = item["offset"], item["tags"]
        tokens = []
        for i, label_id in enumerate(tags):
            #print("---ITEM---")
            #print(item)

            label = id2label[label_id]
            #print(label)
            tag = label[2:] # "I-PER" => "PER"
            if label.startswith("B-"):
                predicted_offsets[tag].append({"start": start, "end": end})
            elif label.startswith("I-"):
                # If "B-" and "I-" both appear in the same tag, ignore as we already processed it
                if label2id[f"B-{tag}"] in tags:
                    continue
                #print("LAST TOKEN TAGS **************")
                #print(last_token_tags)
                if label_id not in last_token_tags and label2id[f"B-{tag}"] not in last_token_tags:
                    predicted_offsets[tag].append({"start": start, "end": end})
                else:
                    predicted_offsets[tag][-1]["end"] = end
        
        last_token_tags = tags
        
        flatten_predicted_offsets = [{**v, "tag": k, "text": example[v["start"]:v["end"]]} 
                                     for k, v_list in predicted_offsets.items() for v in v_list if v["end"] - v["start"] >= 3]
        #print("---- FLATTEEEEEEEEEEEEEEEN -------- ")
        #print(flatten_predicted_offsets)
        flatten_predicted_offsets = sorted(flatten_predicted_offsets, 
                                           key = lambda row: (row["start"], row["end"], row["tag"]))
        merge = merge_groups(flatten_predicted_offsets)
    return merge
    
    
#a veces ocurren cosas como que Pek es una entidad en lugar de Pekín, pero detecta Pekín también.
def remove_wrong_ann(annotations):
    new_ann = []
    for i, a in enumerate(annotations):
        found = False
        #solo para anotaciones de una palabra
        if len(a[-1].split(" ")) == 1:
            j = 0
            while not found and j < len(annotations):
                checking = annotations[j]
                if len(checking[-1].split(" ")) == 1 and i != j:
                    if a[3] >= checking[3] and a[4] <= checking[4] and a[-1] != checking[-1]:
                        found = True
                        """print("---- FOUND ---")
                        print(a)
                        print(checking)"""
                j += 1
        if not found:
            new_ann.append(a)
    return new_ann
    

PATH_TXT = "./data/test_set/txt/"
ann_results = []
total_results = []
for filename in os.listdir(PATH_TXT):
        print("------ filename ---------")
        print(filename)
    #if filename == "casos_clinicos_infecciosas48.txt":
        with open(PATH_TXT + filename, "r") as f:
            t = 0
            text = f.read()
            sentences = text.split(".")
            for s in sentences:
                #for item in get_offsets_and_predicted_tags(s, model, tokenizer):
                #    print(f"""{item["token"]:15} - {item["tags"]}""")
                try:
                    #print(s)
                    #print("--------- BUSCAR FRASE EN TOTAL ----------- ")
                    #print(text.find(s))
                    results = get_tagged_groups(s, model, tokenizer)
                    #print("----------- resultados----------")
                    #print(results)
                    if len(results) > 0:
                        #print(results)
                        for r in results:
                            start = r["start"]
                            end = r["end"]
                            i_s = text.find(s)
                            real_start = i_s + start
                            real_end = i_s + end
                            """print("------ LO QUE SALE ----------")
                            print(text[real_start:real_end])"""
                            ann_results.append((filename.split(".")[0], t, r["tag"], real_start, real_end, r["text"]))
                            t += 1
                except Exception as e:
                    print(e)
                    print("Sentence exceeds length")
                    
                    
with open("august_solution2.tsv", "w") as f:
    f.write("filename\tann_id\tlabel\tstart_span\tend_span\ttext\n")
    for ann in ann_results:
        count = 0
        f.write(ann[0] +"\tT" + str(ann[1]) + "\t" + str(ann[2]) + "\t" + str(ann[3]) + "\t" + str(ann[4]) + "\t" + ann[5] + "\n")
        count += 1
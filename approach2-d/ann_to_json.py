import gzip
import json

def search_in_fragment(split_text, word, start, end):
    i = 0
    discovered = 0
    found = False
    count_letters_begin = 0
    count_letters_end = 0
    while i < len(split_text) and not found:
    
        count_letters_end = count_letters_begin + len(split_text[i])
        if count_letters_begin <= start <= count_letters_end:
            found = True
            discovered = i
            new_start = split_text[i].find(word)
            new_end = new_start + len(word)
        count_letters_begin = count_letters_end + 1
        i += 1
    return discovered, new_start, new_end
    
def get_tags(txt_path, ann_path):
    with open(txt_path, "r", encoding='utf-8-sig') as f:
        text = f.read()
    
    split_text = text.split(".")
    # Read in the annotations from the .ann file and convert them to IOB format
    tags = []
    dict_all_file_results = {}
    file_d = {"tags":"", "id": "", "text":""}
    file_d["id"] = ann_path.split("/")[-1].split(".")[0]
    file_d["text"] = text
    with open(ann_path, "r", encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line.startswith("T"):
                # Extract the entity type and the start and end positions of the entity
                _, data, entity_text = line.split("\t")
                entity_type, start, end = data.split(" ")
                start = int(start)
                end = int(end)
                fragment, new_start, new_end = search_in_fragment(split_text, entity_text, start, end)
                d = {"end": new_end, "start": new_start, "tag": entity_type}
                if fragment in dict_all_file_results:
                    dict_all_file_results[fragment]["tags"].append(d)
                else:
                    dict_all_file_results[fragment] = {"tags": [d], "id": ann_path.split("/")[-1].split(".")[0] , "text":split_text[fragment]}
                
    file_results = []
    for k, v in dict_all_file_results.items():
        file_results.append(v)
        
    return file_results 

def dicts_to_jsonl(data_list: list, filename: str, compress: bool = False) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = '.jsonlines'
    sgz = '.gz'
    # Check filename
    if not filename.endswith(sjsonl):
        filename = filename + sjsonl
    # Save data
    
    if compress:
        filename = filename + sgz
        with gzip.open(filename, 'w') as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(filename, 'w', encoding="utf-8") as out:
            for ddict in data_list:
                jout = json.dumps(ddict, ensure_ascii=False).encode('utf8')
                jout = jout.decode() + "\n"
                out.write(jout)
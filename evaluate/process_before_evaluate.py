import json
import re

prediction_file = "/data0/luohaibo/llm/predictions/NER.json"
golden_truth_file = "/data0/luohaibo/llm/data/format_genia_test.json"  # may be modified
output_file = "/data0/luohaibo/llm/evaluate/data/evaluate.json"

data_list = []
output_list = []
result_list = []

with open(golden_truth_file, "r", encoding='utf-8') as file:
    for line in file:
        data_list.append(json.loads(line))

with open(prediction_file, "r", encoding="utf-8") as file:
    data = json.load(file)
    for item in data:
        output_text = item["Output"].replace(item["Input"], '')
        output_list.append(output_text)

# compute the length of previous sentences before current sentence
def get_length_before_index(lst, index): 
    length = 0
    for i in range(index):
        # print(f'current length:{length}, current sentence length:{len(lst[i])}')
        length = length + len(lst[i])
    return length

output_list_index = -1

for data in data_list:
    for index, word_list in enumerate(data['sentences']):

        output_list_index = output_list_index + 1

        # 1.generate sentence
        sentence = ' '.join(word_list)

        # 2.generate gold_pair
        gold_pair_list = []
        predict_pair_list = []

        length = get_length_before_index(data['sentences'], index)
        ner_list = data['ner'][index]
        # print(ner_list)
        for ner in ner_list:
            # print(ner)
            start_index = ner[0] - length
            end_index = ner[1] - length + 1
            # print(
            #     f'sentence length:{len(sentence)}, previous index:{ner}, previous length:{length}, new index{start_index, end_index}')
            entity_name = word_list[start_index:end_index]
            entity_type = ner[2]
            entity_name = ' '.join(entity_name)

            temp_gold_pair_dict = {'entity_name': entity_name, 'entity_type': entity_type}
            # print(temp_gold_pair_dict)
            gold_pair_list.append(temp_gold_pair_dict)

        # 3.generate predict_pair
        pattern = r'\{([^}]*)\}'
        # print(output_list[index])
        # print(index)
        matches = re.findall(pattern, output_list[output_list_index])

        for match in matches:
            inner_text = match.strip()
            inner_elements = re.findall(r'<(.*?)>', inner_text)
            if len(inner_elements) == 2:
                temp_predict_pair_dict = {'entity_name': inner_elements[0],
                                          'entity_type': inner_elements[1]}
                predict_pair_list.append(temp_predict_pair_dict)
        # print(predict_pair_list)

        # 4.generate result
        temp_result_dict  = {'text':sentence, 'gold_pair':gold_pair_list, 'predict_pair':predict_pair_list}
        result_list.append(temp_result_dict)

# write result
with open(output_file, "w", encoding='utf-8') as file:
    json.dump(result_list, file, indent=4)


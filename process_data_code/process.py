import json
import random

source_file = "/data0/luohaibo/llm/data/format_genia_train_dev.json"  # may be modified
output_file = "/data0/luohaibo/llm/processed_data/train_dev.json"

def get_length_before_index(lst, index):
    length = 0
    for i in range(index):
        print(f'current length:{length}, current sentence length:{len(lst[i])}')
        length = length + len(lst[i])
    return length


data_list = []
with open(source_file, "r", encoding='utf-8') as file:
    for line in file:
        data_list.append(json.loads(line))

result_list = []
count_empty = 0
for data in data_list:
    for index, sentence in enumerate(data['sentences']):
        entity_list = ''
        entity_name_list = ''
        length = get_length_before_index(data['sentences'], index)
        ner_list = data['ner'][index]
        # print(ner_list)
        for ner in ner_list:
            # print(ner)
            start_index = ner[0] - length
            end_index = ner[1] - length + 1
            print(f'sentence length:{len(sentence)}, previous index:{ner}, previous length:{length}, new index{start_index, end_index}')
            entity_name = sentence[start_index:end_index]
            entity_type = ner[2]
            entity_name = ' '.join(entity_name)
            # print(entity_name)
            entity_name_list = entity_name_list + '{<' + entity_name + '>}, '
            entity_list = entity_list + '{<' + entity_name + '>, <' + entity_type + '>}, '
        entity_list = entity_list[:-2]
        entity_name_list = entity_name_list[:-2]
        if entity_list == '':
            entity_list = 'There is no information to be extracted from the sentence.'
            count_empty = count_empty + 1
            # random_number = random.random()
            # if random_number <= 0.8:
            #     continue
        result = {'instruction': 'Find all the entities in the sentence and classify them. Format:{<entity>, <entity type>}. ',
                  'input': ' '.join(sentence),
                  'output': entity_list}
        result_list.append(result)

        if entity_name_list != '':
            result = {
                'instruction': 'Find all the entities in the sentence. Format:{<entity>}. ',
                'input': ' '.join(sentence),
                'output': entity_name_list}
            result_list.append(result)
            result = {
                'instruction': 'Classify the entities in the entity list according to the sentence and entity list. Format:{<entity>, <entity type>}. ',
                'input': ' '.join(sentence) + ' Entity list: ' + entity_name_list,
                'output': entity_list}
            result_list.append(result)
        else:
            result = {
                'instruction': 'Find all the entities in the sentence. Format:{<entity>}. ',
                'input': ' '.join(sentence),
                'output': 'There is no information to be extracted from the sentence.'}
            result_list.append(result)

print(f'number of empty:{count_empty}')
print(f'length_of_result_list:{len(result_list)}')

with open(output_file, "w", encoding='utf-8') as file:
    json.dump(result_list, file, indent=4)


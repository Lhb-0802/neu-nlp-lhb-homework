import json

source_file = "/data0/luohaibo/llm/data/format_genia_test.json"  # may be modified
output_file = "/data0/luohaibo/llm/processed_data/test.txt"

data_list = []
with open(source_file, "r", encoding='utf-8') as file:
    for line in file:
        data_list.append(json.loads(line))

result_list = []
for data in data_list:
    for index, sentence in enumerate(data['sentences']):
        result = {'instruction': 'Find all the entities in the sentence and classify them. Format:{<entity>, <entity type>}. ',
                  'input': ' '.join(sentence)}
        result_list.append(str(result['instruction'] + result['input']))

print(f'length_of_result_list:{len(result_list)}')

with open(output_file, "w", encoding='utf-8') as file:
    for result in result_list:
        file.write(result + '\n')

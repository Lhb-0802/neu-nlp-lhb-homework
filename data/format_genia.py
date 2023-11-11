import json

# source_file = "/data0/luohaibo/llm/data/genia_train_dev.json"
# output_file = "/data0/luohaibo/llm/data/format_genia_train_dev.json"
source_file = "/data0/luohaibo/llm/data/genia_test.json"
output_file = "/data0/luohaibo/llm/data/format_genia_test.json"

result_list = []

with open(source_file, "r", encoding='utf-8') as file:
    data = file.read()  # 从文件中读取数据
    data_list = json.loads(data)  # 解析JSON数据

for data in data_list:
    try:
        tokens = data['tokens']
        entities = data['entities']
        entity_list = []
        for entity in entities:
            entity_list.append([entity['start'], (entity['end'] - 1), entity['type']])
        result_dict = {'sentences': [tokens], 'ner': [entity_list]}
        result_list.append(result_dict)
    except IOError as e:
        print(f"An error occurred: {e}")

with open(output_file, "w", encoding='utf-8') as file:
    for result in result_list:
        json_str = json.dumps(result)
        file.write(json_str + '\n')

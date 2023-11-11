import json

evaluate_file = '/data0/luohaibo/llm/evaluate/data/evaluate.json'  # may be modified

with open(evaluate_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

def is_dict_in_list(target_dict, dict_list):
    for dictionary in dict_list:
        if dictionary['entity_name'] == target_dict['entity_name']:
            if dictionary['entity_type'] == target_dict['entity_type']:
                return True
    return False


count_predict_pair = 0
count_true_pair = 0
count_pair_true = 0
count_head_true = 0
count_pair_false = 0
count_predict_no_entity = 0
count_true_no_entity = 0
count_no_entity_true = 0

for line in data:
    text = line['text']

    count_true_pair = count_true_pair + len(line['gold_pair'])
    count_predict_pair = count_predict_pair + len(line['predict_pair'])

    if len(line['gold_pair']) == 0:
        count_true_no_entity = count_no_entity_true + 1
    if len(line['predict_pair']) == 0:
        count_predict_no_entity = count_predict_no_entity + 1
    if len(line['gold_pair']) == 0 and len(line['predict_pair']) == 0:
        count_no_entity_true = count_no_entity_true + 1

    for relation_pair in line['predict_pair']:
        if relation_pair['entity_name'] in text:
            count_head_true = count_head_true + 1

        if is_dict_in_list(relation_pair, line['gold_pair']):
            count_pair_true = count_pair_true + 1
        else:
            count_pair_false = count_pair_false + 1


print(f'实体对（entity pair）的真值数量：{count_true_pair}')
print(f'预测的数量：{count_predict_pair}')
print(f'预测正确的数量：{count_pair_true}')
print(f'预测错误的数量：{count_pair_false}')
print(f'预测的实体是句子的子串：{count_head_true}')
print(f'句子不存在实体的真值数量：{count_true_no_entity}')
print(f'预测句子不存在实体的数量：{count_predict_no_entity}')
print(f'预测正确的数量：{count_no_entity_true}')


precision = count_pair_true / count_predict_pair
recall = count_pair_true / count_true_pair
f1 = 0
if precision !=0 and recall != 0:
    f1 = 2 * precision * recall / (precision + recall)
print(f'precision:{precision}, recall:{recall}, f1:{f1}')

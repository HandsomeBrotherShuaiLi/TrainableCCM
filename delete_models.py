import os
def delete_earier_models():
    level_1=os.listdir('train')
    level_2 = os.listdir(os.path.join('train','epoch'))
    name_set=set()
    for i in level_1:
        if i.startswith('checkpoint-'):
            name_set.add(i.split('.')[0])
    name_list = sorted(list(name_set))[:-2]
    if len(name_list)>=3:
        for i in level_1:
            for j in name_list:
                if i.startswith(j):
                    try:
                        os.remove(os.path.join('train', i))
                        print("删除{}模型".format(os.path.join('train', i)))
                    except Exception as e:
                        print(e)
    name_set = set()
    for i in level_2:
        if i.startswith('checkpoint-'):
            name_set.add(i.split('.')[0])
    name_list = sorted(list(name_set))[:-2]
    if len(name_list)>=3:
        for i in level_2:
            for j in name_list:
                if i.startswith(j):
                    try:
                        os.remove(os.path.join('train','epoch', i))
                        print("删除{}模型".format(os.path.join('train','epoch',i)))
                    except Exception as e:
                        print(e)
delete_earier_models()

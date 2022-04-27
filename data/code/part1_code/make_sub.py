import pandas as pd
import numpy as np

sub = pd.read_pickle('/data/user_data/infer/oof_vq2.pkl')
sub['label'] = sub['image_path'].apply(lambda x: x.split('/')[-1][:-4])

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)

total = sub.drop(['image_path', 'label'], axis=1).values.reshape(-1, 62)
total = np.argmax(total, axis=1)
total = total.reshape(-1, 4)
res = []
for o in total:
    o = ''.join([alphabet[i] for i in o])
    res.append(o)


def calculat_acc(output, target):
    count = 0
    for i in range(len(target)):
        if output[i] == target[i]:
            count += 1
    acc = count / len(target)
    return acc


for i in range(5):
    print(f"Fold {i}:", calculat_acc(res[3000 * i:3000 * (i + 1)], sub['label'].values[3000 * i:3000 * (i + 1)]))

print(f"Fold mean:", calculat_acc(res, sub['label'].values))


sub = pd.read_pickle('/data/user_data/infer/oof_v4.pkl')
sub2 = pd.read_pickle('/data/user_data/infer/oof_v6.pkl')
sub3 = pd.read_pickle('/data/user_data/infer/oof_v7.pkl')
sub4 = pd.read_pickle('/data/user_data/infer/oof_vq1.pkl')
sub5 = pd.read_pickle('/data/user_data/infer/oof_vq2.pkl')
sub6 = pd.read_pickle('/data/user_data/infer/oof_vq3.pkl')

sub['label'] = sub['image_path'].apply(lambda x:x.split('/')[-1][:-4])


# sub7 = pd.read_pickle('infer/oof_lh.pkl')
sub7 = pd.read_csv('/data/user_data/infer/oof_lh.csv')
sub7['label'] = sub7['label'].apply(lambda x: x[-4:])
sub7 = pd.merge(sub[['label']], sub7, on='label', how='left')
del sub7['img_path'], sub7['label'], sub7['fold']

prob1 = sub.drop(['image_path', 'label'], axis=1).values
prob2 = sub2.drop(['image_path'], axis=1).values
prob3 = sub3.drop(['image_path'], axis=1).values
prob4 = sub4.drop(['image_path'], axis=1).values
prob5 = sub5.drop(['image_path'], axis=1).values
prob6 = sub6.drop(['image_path'], axis=1).values
prob7 = sub7.values

import optuna


def objective(trial):
    model_weights = [trial.suggest_uniform(f'oof_weights_{x}', 0, 1) for x in range(7)]
    model_weights = [i / sum(model_weights) for i in model_weights]
    probs_merge = prob1 * model_weights[0] + prob2 * model_weights[1] + prob3 * model_weights[2] + prob4 * \
                  model_weights[3] + \
                  prob5 * model_weights[4] + prob6 * model_weights[5] + prob7 * model_weights[6]
    probs_merge = probs_merge.reshape(-1, 62)
    total = np.argmax(probs_merge, axis=1)
    total = total.reshape(-1, 4)
    res = []
    for o in total:
        o = ''.join([alphabet[i] for i in o])
        res.append(o)
    score = calculat_acc(res, sub['label'].values)
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)


model_weights = [study.best_params[f'oof_weights_{x}'] for x in range(7)]    # 0.98973

model_weights = [i / sum(model_weights) for i in model_weights]

sub1 = pd.read_pickle('/data/user_data/infer/predict_v4.pkl').values * model_weights[0]
sub2 = pd.read_pickle('/data/user_data/infer/predict_v6.pkl').values * model_weights[1]
sub3 = pd.read_pickle('/data/user_data/infer/predict_v7.pkl').values * model_weights[2]
sub4 = pd.read_pickle('/data/user_data/infer/predict_vq1.pkl').values * model_weights[3]
sub5 = pd.read_pickle('/data/user_data/infer/predict_vq2.pkl').values * model_weights[4]
sub6 = pd.read_pickle('/data/user_data/infer/predict_vq3.pkl').values * model_weights[5]
sub7 = pd.read_csv('/data/user_data/infer/predict_lh.csv').values * model_weights[6]

sub = sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7
source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)

total = sub.reshape(-1, 62)
total = np.argmax(total, axis=1)
total = total.reshape(-1, 4)
res = []
for o in total:
    o = ''.join([alphabet[i] for i in o])
    res.append(o)

result = pd.DataFrame({'num':[i for i in range(1, 15001)], 'tag':res})
result.to_csv('/data/prediction_result/sub_final.csv', index=False)
result.head()
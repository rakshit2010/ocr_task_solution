# evaluate.py
import json, os
from sklearn.metrics import accuracy_score

# Expect a JSON file with list of records: {'image': 'name', 'gt':'ground_truth', 'predicted':'pred'}
def load_records(path):
    return json.load(open(path, 'r', encoding='utf-8'))

def exact_accuracy(records):
    y_true = [r['gt'] for r in records]
    y_pred = [r.get('predicted') or r.get('extracted') for r in records]
    correct = [1 if a==b else 0 for a,b in zip(y_true,y_pred)]
    acc = sum(correct)/len(correct) if correct else 0.0
    return acc

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--labels', required=True, help='Path to labels JSON (list of {image,gt})')
    p.add_argument('--preds', required=True, help='Path to predictions JSON (list of {image,predicted})')
    args = p.parse_args()
    labels = load_records(args.labels)
    preds = {r['image']: r for r in load_records(args.preds)}
    merged = []
    for item in labels:
        img = item['image']
        gt = item['gt']
        pred = preds.get(img, {}).get('predicted') or preds.get(img, {}).get('extracted')
        merged.append({'image': img, 'gt': gt, 'predicted': pred})
    acc = exact_accuracy(merged)
    print('Exact match accuracy:', acc)

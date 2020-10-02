import argparse
import json
import pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='')
    parser.add_argument('output', help='')
    args = parser.parse_args()

    sent_ids_file = args.output[:-6] + '-sent-ids.jsonl'
    indexes = []
    with open(args.input) as fin, open(args.output, 'w') as fout, open(sent_ids_file, 'w') as fout2:
            for line in fin:
                obj = json.loads(line)
                new_obj = {'text': obj['source'],
                            'summary': [obj['target'][0]] if isinstance(obj['target'], list) else [obj['target']],
                            'summary_id': obj['paper_id']}
                fout.write(json.dumps(new_obj))
                fout.write('\n')
                sent_ids = {"sent_id": [i for i in range(len(obj['source']))]}
                fout2.write(json.dumps(sent_ids))
                fout2.write('\n')

if __name__ == '__main__':
    main()

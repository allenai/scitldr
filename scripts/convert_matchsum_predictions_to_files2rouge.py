import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='')
    parser.add_argument('output', help='')
    args = parser.parse_args()

    ref = []
    for f in glob.glob(args.input + '/**/*.ref', recursive=True):
        with open(f) as fin:
            ref.append(fin.read().strip())
    res=[]
    i = 0
    for f in glob.glob(args.input + '/**/*.dec', recursive=True):
        with open(f) as fin:
            res.append(fin.read().strip())
    with open(args.output, 'w') as fout:
        for e in res:
            fout.write(e.strip().replace('\n',' '))
            fout.write('\n')

if __name__ == '__main__':
    main()

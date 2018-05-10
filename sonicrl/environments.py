import csv


def get_environments(path):
    with open(path) as f:
        environments = list(csv.DictReader(f))
        for env in environments:
            env['scenario'] = 'contest'
        return environments

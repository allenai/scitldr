"""

Some analysis of informational content of TLDR-Auth and TLDR-PR

"""

import os
import csv

from collections import Counter, defaultdict

INFILE = 'tldr_analyze_nuggets/tldr_auth_pr_gold_nuggets_2020-03-31.csv'

# Q1: How many nuggets do TLDRs contain?
# A:  Interesting, both author and PR have nearly identical distributions:
#     From most to least common:  3 nuggets -> 2 nuggets -> 4 nuggets -> 1 nugget -> ...
#              Auth proportions:    (34%)        (26%)        (18%)         (11%)
#              PR   proportions:    (32%)        (30%)        (26%)         ( 9%)
author_num_nuggets_to_count = {i: 0 for i in range(0,7)}
pr_num_nuggets_to_count = {i: 0 for i in range(0,7)}
with open(INFILE) as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        num_nuggets = sum(map(int, [row['area_field_topic'], row['problem_motivation'], row['mode_of_contrib'], row['details_descrip'], row['results_findings'], row['value_signif']]))
        if row['auth_pr'] == 'auth_gold':
            author_num_nuggets_to_count[num_nuggets] += 1
        if row['auth_pr'] == 'pr_gold':
            pr_num_nuggets_to_count[num_nuggets] += 1
print({k: f'{100*v/76:.2f}' for k, v in author_num_nuggets_to_count.items()})
print({k: f'{100*v/76:.2f}' for k, v in pr_num_nuggets_to_count.items()})


# Q2: What are the most common TLDR templates?
# A:  Interesting, the top 2 templates (total 42 occurrences) are same between Authors and PRs.
#       a) (area_field_topic, mode_of_contrib, details_descrip)
#       b) (area_field_topic, mode_of_contrib)
#     After that, next 3 starts deviating a bit, but still with the same base:
#       authors = (area_field_topic, mode_of_contrib, results_findings)
#                 (area_field_topic, problem_motivation, mode_of_contrib)
#                 (area_field_topic, mode_of_contrib, details_descrip, value_signif)
#       pr      = (area_field_topic, problem_motivation, mode_of_contrib, details_descrip)
#               = (area_field_topic, details_descrip)
#               = (area_field_topic, mode_of_contrib, results_findings)     # same as top 3rd in Auth
author_template_to_count = Counter()
pr_template_to_count = Counter()
with open(INFILE) as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        template = (row['area_field_topic'], row['problem_motivation'], row['mode_of_contrib'], row['details_descrip'], row['results_findings'], row['value_signif'])
        if row['auth_pr'] == 'auth_gold':
            author_template_to_count[template] += 1
        if row['auth_pr'] == 'pr_gold':
            pr_template_to_count[template] += 1
print(author_template_to_count.most_common())
print(pr_template_to_count.most_common())


# Q3:  How often does 'area_field_topic' and 'mode_of_contrib' co-occur?
#      n_auth = 48/76 = 63%
#      n_pr = 54/76 = 71%
n_auth = 0
n_pr = 0
with open(INFILE) as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        if row['area_field_topic'] == '1' and row['mode_of_contrib'] == '1':
            if row['auth_pr'] == 'auth_gold':
                n_auth += 1
            if row['auth_pr'] == 'pr_gold':
                n_pr += 1


# Q4:  Find examples with exactly the same nuggets but different styles
#
# H1-IBSgMz
# B16yEqkCZ
# SySpa-Z0Z
# rJegl2C9K7
# HJWpQCa7z
# rkgpCoRctm
# rkxkHnA5tX
# B1e9csRcFm
# r1kj4ACp-
# Hk91SGWR-
# r1GaAjRcF7
# SkGMOi05FQ
#
pid_to_templates = defaultdict(set)
with open(INFILE) as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        template = (row['area_field_topic'], row['problem_motivation'], row['mode_of_contrib'], row['details_descrip'], row['results_findings'], row['value_signif'])
        pid_to_templates[row['paper_id']].add(template)
for pid, templates in pid_to_templates.items():
    if len(templates) == 1:
        print(pid)

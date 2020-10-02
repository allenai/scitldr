rm(list=ls())
setwd('tldr_pred_rouge_sig_testing')

# copied from `significance_tests.csv` file:

if(FALSE){
  # AO -- \bart vs \ours
  R1	multi.max	ao	....bart.large.20k.titles.20k.tldr..ctrl.	bart.large.tldr	0.47441	0.47804	-0.46438	1.4132	0.321391
  R2	multi.max	ao	....bart.large.20k.titles.20k.tldr..ctrl.	bart.large.tldr	0.14218	0.54798	-0.93395	1.21831	0.795366
  RL	multi.max	ao	....bart.large.20k.titles.20k.tldr..ctrl.	bart.large.tldr	0.51901	0.51676	-0.49581	1.53384	0.315597
  
  # AO -- \bartxsum vs \oursxsum
  R1	multi.max	ao	....bart.large.xsum.20k.titles.20k.tldr..ctrl.	bart.large.xsum.tldr	1.76121	0.4922	0.79461	2.72781	0.000373
  R2	multi.max	ao	....bart.large.xsum.20k.titles.20k.tldr..ctrl.	bart.large.xsum.tldr	0.22329	0.5339	-0.8252	1.27177	0.675932
  RL	multi.max	ao	....bart.large.xsum.20k.titles.20k.tldr..ctrl.	bart.large.xsum.tldr	0.96877	0.52834	-0.06879	2.00633	0.067193
  
  # AIC -- \bart vs \ours
  R1	multi.max	aic	....bart.large.20k.titles.20k.tldr..ctrl.	bart.large.tldr	2.05745	0.50677	1.06224	3.05266	5.50E-05
  R2	multi.max	aic	....bart.large.20k.titles.20k.tldr..ctrl.	bart.large.tldr	1.79829	0.58783	0.6439	2.95268	0.002315
  RL	multi.max	aic	....bart.large.20k.titles.20k.tldr..ctrl.	bart.large.tldr	2.18028	0.54399	1.11199	3.24857	6.90E-05
  
  # AIC -- \bartxsum vs \ourxsum
  R1	multi.max	aic	....bart.large.xsum.20k.titles.20k.tldr..ctrl.	bart.large.xsum.tldr	0.98418	0.47321	0.05488	1.91348	0.037956
  R2	multi.max	aic	....bart.large.xsum.20k.titles.20k.tldr..ctrl.	bart.large.xsum.tldr	0.36686	0.54387	-0.70119	1.43492	0.500215
  RL	multi.max	aic	....bart.large.xsum.20k.titles.20k.tldr..ctrl.	bart.large.xsum.tldr	0.51317	0.5091	-0.48661	1.51295	0.313852
}

# perform Holm-Bonferroni correction within each comparison class of metrics

# post-adjustment, 0.946791 0.946791 0.946791   -->  AO  \bart vs \ours, no difference
pvalues = c(0.321391, 0.795366, 0.315597)
p.adjust(p=pvalues, method='holm')

# post-adjustment, 0.001119 0.675932 0.134386   -->  AO  \bartxsum vs \oursxsum, significant R1
pvalues = c(0.000373, 0.675932, 0.067193)
p.adjust(p=pvalues, method='holm')

# post-adjustment, 0.000165 0.002315 0.000165   -->  AIC \bart vs \ours, significant R1, R2, R3
pvalues = c(5.50E-05, 0.002315, 6.90E-05)
p.adjust(p=pvalues, method='holm')

# post-adjustment, 0.113868 0.627704 0.627704   -->  AIC \bartxsum vs \oursxsum, no difference; R1 became not significant :(
pvalues = c(0.037956, 0.500215, 0.313852)
p.adjust(p=pvalues, method='holm')




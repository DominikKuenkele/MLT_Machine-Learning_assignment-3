# Explore and Experiment
## Results
Precision, Recall and F1-Score are calculated, using the macro average.

For the perplexity, I used the first 10.000 lines of the UN corpus.
I didn't use random sentences out of the corpus to get somewhat deterministic results.

### Multinomial Naive Bayes
| sample size | Accuracy | Precision | Recall  | F1-Score | Perplexity |
|-------------|----------|-----------|---------|----------|------------|
| **1.000**   | 0.19     | 0.08961   | 0.08340 | 0.07069  | 17.49      |
| **5.000**   | 0.203    | 0.11524   | 0.08141 | 0.07616  | 14.21      |
| **10.000**  | 0.2195   | 0.07830   | 0.08410 | 0.07100  | 13.26      |
| **20.000**  | 0.20775  | 0.09205   | 0.08675 | 0.07153  | 12.78      |
| **50.000**  | 0.2099   | 0.15851   | 0.11291 | 0.10529  | 12.28      |
| **100.000** | 0.222    | 0.15877   | 0.11776 | 0.11027  | 12.02      |
| **500.000** | 0.22045  | 0.16720   | 0.11594 | 0.10867  | 11.81      |


### Support Vector Classifier
The training with 500000 samples took too long, so there are only scores for models
trained on up to 100000 samples.

| sample size | Accuracy | Precision | Recall  | F1-Score | Perplexity |
|-------------|----------|-----------|---------|----------|------------|
| **1.000**   | 0.195    | 0.13708   | 0.09815 | 0.10261  | 13.15      |
| **5.000**   | 0.215    | 0.08966   | 0.08493 | 0.07303  | 12.45      |
| **10.000**  | 0.218    | 0.11291   | 0.09495 | 0.08661  | 12.28      |
| **20.000**  | 0.21525  | 0.09664   | 0.08463 | 0.07008  | 12.21      |
| **50.000**  | 0.2088   | 0.12640   | 0.07565 | 0.06159  | 12.20      |
| **100.000** | 0.221    | 0.13763   | 0.07793 | 0.06446  | 12.21      |
| **500.000** | -        | -         | -       | -        | -          |


## Discussion
### Training time
The training times of both methods are very different. The multinomial NB learns very fast, independent of
the sample size. On my computer, it took only a few seconds. The SVM on the other hand depends very much on
the sample size. While 1000 training samples took also only a few seconds, the training with 100.000 samples
lasted for 3,5 hours.

### Performance
Comparing the performance over different sample sizes is difficult, since the samples a generated randomly.
This means, that there can be "lucky" sample creations, which perform just better in training/tests. Looking at the
scores, this seemed to happen especially for the sample size of 5000.
Apart from that, it can be seen that all scores get gradually better for the multinomial NB, when increasing the sample size.
Especially the perplexity decreases a lot for the first few increases of the sample size.

For the SVM, the results look somewhat differently. While the accuracy and perplexity increase visibly with growing
sample size, the other scores behave different. The precision seems also to get better with more samples, but this 
is not as clear as for the multinomial NB. Reasons for that may also lie on the random sample generation, as mentioned before.
The recall (and therefore the f1 score) on the other hand is clearly getting worse, when increasing the sample size.

Looking at the accuracy, both methods perform very similar. But for the precision and recall (and f1 score), the multinomial
NB performs better, especially when using many samples for training and testing.
The SVM can outperform the multinomial NB in the perplexity with only few samples. Increasing the sample size, both methods
perform more and more similar. For a very high sample size, the multinomial NB even surpasses the SVM.
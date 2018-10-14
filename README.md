# DLKNLP
### Day-by-day Line-by-line Keras-based Korean NLP
## Sentence classification: From data construction to BiLSTM self-attention

## Contents (to be updated)
0. Corpus labelling
1. Data preprocessing
2. One-hot encoding and basic classifiers
3. TF-IDF and basic classifiers
4. NN classifier using Keras
5. Dense word vector embedding and Document vectors
6. CNN-based sentence classification
7. RNN (BiLSTM)-based sentence classification
8. Concatenation of CNN and RNN layers
9. BiLSTM Self-attention

## 0. Corpus labeling
The most annoying and confusing process.
Annotation guideline should be provided to annotators and more than two natives should be engaged in to make the labeling reliable and also for a computation of inter-annotator agreement (IAA). In this project, multi-class (7) annotation of short Korean utterances is utilized.
* 데이터를 만드는, 가장 귀찮은 과정입니다. 언어학적 직관은 1인분이기 때문에, 레이블링이 설득력을 얻기 위해서는 적어도 3명 이상의 1화자를 통한 레이블링으로 그 타당성을 검증해야 합니다 (아카데믹하게는...) 본 프로젝트에서는 7-class의 한국어 문장들이 분류에 사용됩니다.

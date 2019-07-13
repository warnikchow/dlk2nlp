# DLK2NLP: Day-by-day Line-by-line Keras-based Korean NLP
## Sentence classification: From data construction to self-attentive BiLSTM
### The tutorial is provided in both English (for *Korean* NLP learners) and Korean (for Korean *NLP* learners) - but the meanings may differ.
* 문장 분류 task를 중심으로 본 한국어 NLP 튜토리얼입니다. 본 튜토리얼에 사용된 컨텐츠 일부는 [페이스북 페이지](https://www.facebook.com/nobodybelongs/notes/)에 기고했던 글에서 발췌하였음을 명시합니다. 한국어 NLP를 하고 싶은 외국인들과 NLP를 배워보고 싶은 한국인들을 모두 대상으로 하여 영/한 설명을 모두 제공하나, 번역이 아닌 의역이며 내용도 다를 수 있습니다.

## Requirements
#### The recommended versions are in *Requirements.txt*, but can be replaced depending on the environment
fasttext (Gensim if inavailable), Keras, konlpy (refer to the [documentation](http://konlpy.org/en/v0.4.4/)), hgtk, nltk, numpy, scikit-learn, tensorflow-gpu
#### Download [100-dimension fastText vector dictionary which was trained with 2M drama scripts](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor) and place in the folder named *vectors*. For the utilization of the dictionary, cite the following:
```
@article{cho2018real,
	title={Real-time Automatic Word Segmentation for User-generated Text},
	author={Cho, Won Ik and Cheon, Sung Jun and Kang, Woo Hyun and Kim, Ji Won and Kim, Nam Soo},
	journal={arXiv preprint arXiv:1810.13113},
	year={2018}
}
```
#### Download the [dataset](https://github.com/warnikchow/3i4k/blob/master/data/fci.txt) and place in the folder named *data*. For the utilization of the dataset, cite the following:
```
@article{cho2018speech,
	title={Speech Intention Understanding in a Head-final Language: A Disambiguation Utilizing Intonation-dependency},
	author={Cho, Won Ik and Lee, Hyeon Seung and Yoon, Ji Won and Kim, Seok Min and Kim, Nam Soo},
	journal={arXiv preprint arXiv:1811.04231},
	year={2018}
}
```

## Contents (to be updated)
[0. Corpus labelling](https://github.com/warnikchow/dlk2nlp/blob/master/README.md#0-corpus-labeling)</br>
[1. Data preprocessing](https://github.com/warnikchow/dlk2nlp#1-data-preprocessing)</br>
[2. One-hot encoding and sentence vector](https://github.com/warnikchow/dlk2nlp#2-one-hot-encoding-and-sentence-vector)</br>
[3. TF-IDF and basic classifiers](https://github.com/warnikchow/dlk2nlp#3-tf-idf-and-basic-classifiers)</br>
[4. Dense word embeddings](https://github.com/warnikchow/dlk2nlp#4-dense-word-embeddings)</br>
[5. Document vectors and NN classifier](https://github.com/warnikchow/dlk2nlp#5-document-vectors-and-nn-classifier)</br>
[6. CNN-based sentence classification](https://github.com/warnikchow/dlk2nlp#6-cnn-based-sentence-classification)</br>
[7. RNN (BiLSTM)-based sentence classification](https://github.com/warnikchow/dlk2nlp#7-rnn-bilstm-based-sentence-classification)</br>
[8. Character embedding](https://github.com/warnikchow/dlk2nlp#8-character-embedding)</br>
[9. Concatenation of CNN and RNN layers](https://github.com/warnikchow/dlk2nlp#9-concatenation-of-cnn-and-rnn-layers)</br>
[10. Self-attentive BiLSTM](https://github.com/warnikchow/dlk2nlp#10-self-attentive-bilstm)</br>
[11. Transformer, BERT, and after](https://github.com/warnikchow/dlk2nlp#11-transformer-bert-and-after)

## 0. Corpus labeling
The most annoying and confusing process. Annotation guideline should be provided to annotators, and more than two natives are to be engaged in to make the labeling reliable and also for the computation of inter-annotator agreement (IAA). In this project, a multi-class (7) annotation of short Korean utterances is utilized.

* 데이터를 만드는, 가장 귀찮은 과정입니다. 언어학적 직관은 1인분이기 때문에, 레이블링이 설득력을 얻기 위해서는 적어도 3명 이상의 1화자를 통한 레이블링으로 그 타당성을 검증해야 합니다 (아카데믹하게는...) 
* 본 프로젝트에서는 7-class의 한국어 문장 분류 방법이 사용됩니다.

---

The task is on classification; especially about extracting intention from a single utterance with the punctuation removed, which is suggested in [3i4k](https://github.com/warnikchow/3i4k). As the description displays, the corpus was partially hand-annotated and incorporates the utterances which are generated or semi-automatically collected. The total number of the utterances reaches 61K, with each label denoting</br></br>
**0: Fragments**</br>
**1: Statement**</br>
**2: Question**</br>
**3: Command**</br>
**4: Rhetorical question**</br>
**5: Rhetorical command**</br>
**6: Intonation-dependent utterance**</br></br>
where the [inter-annotator (IAA)](https://en.wikipedia.org/wiki/Cohen%27s_kappa) was computed 0.85 (quite high!) for the manually annotated 2K utterance set (corpus 1 in the table below). The [annotation guideline](https://drive.google.com/open?id=1AvxzEHr7wccMw7LYh0J3Xbx5GLFfcvMW) is in Korean.</br>
<p align="center">
    <image src="https://github.com/warnikchow/3i4k/blob/master/images/portion.PNG" width="400">
    
* 태스크는 의도 분류로써, [3i4k](https://github.com/warnikchow/3i4k) 프로젝트를 위해 제작된 DB를 사용합니다. 사실 국책과제에 쓰려고 만든건데 어차피 논문으로도 submit했으니 공개는 상관 없지 않을까 싶어요. 6만 1천 문장쯤으로 아주 규모가 크지는 않지만, 일단 수작업으로 2만 문장 정도에서 0.85의 IAA를 얻었으며 (꽤 높은 agreement!), 4만 문장 가량이 더 수집/생성되어 그래도 어느정도 쓸만한 데이터셋이 만들어졌습니다. 

* 레이블 7개는 위에 써 둔 것처럼, Statement~Rhetorical command까지의 clear한 의도 5가지와 (논문에선 clear-cut cases라고 칭했습니다만), 의도가 불분명한 Fragment (명사, 명사구, 혹은 불완전한 문장), 마지막으로 Intonation-dependent utterances *억양에 따라 의도가 달라지는 문형* 입니다. 마지막 레이블은 저 논문에서 하나의 문장 유형으로 분류하기로 제안한 것이지만, 한국어 화자라면 어떤 문장들이 그런 성질을 가지는지 감이 올 것입니다. "뭐 먹고 싶어" "천천히 가고 있어" 같은 문장들이 그러한 유형이죠. 주로 puncutation이 제거된 상태로 등장하는 spoken language의 understanding에 아주 골머리를 썩이는 녀석들이기 때문에 따로 분류하기로 하였습니다. 

* Annotation guideline이 어떤 형식인지 궁금하신 분들은 [이곳](https://drive.google.com/open?id=1AvxzEHr7wccMw7LYh0J3Xbx5GLFfcvMW)을 참고하시면 됩니다.

## 1. Data preprocessing
For the next step, we should pay attention to how we can manage HANGUL, the letters of Korean writing system (WS). Korean WS and its alphabet *Jamo* which was (maybe solely) invented by The Great King Sejong, incorporate special morpho-syllabic blocks which each acts as a syllable and consists of CV(C), making Korean as a representative language with featural WS. The blocks are agglutinated to make up a word *eojeol*, which should be decomposed into morphemes for semantically meaningful language processing. Spaces go between the *eojeol*s, to enhance the readability of a sentence. Many morphological analyzers give an additional spacing between the morphemes; some analyzers such as [Twitter](https://github.com/twitter/twitter-korean-text) simply give spaces, and there are the ones which conduct an elaborate tokenizing process of block decomposition, such as [Kkma](http://kkma.snu.ac.kr/). Total five analyzers are wrapped in a famous Korean natural language processing toolkit, [KoNLPy](http://konlpy.org/en/v0.4.4/). In this tutorial, we proceed with the Twitter analyzer, for its speed and to prevent the characters being decomposed.

* 한국어와 한글, 문자 체계, 형태소, 띄어쓰기 및 형태소 분석기에 대해 주절주절 설명해 봤는데요, 아마 한국어 L1 화자라면 대부분 익숙한 내용이실 테니 이 부분의 한글 설명은 생략하도록 하겠습니다 ㅎㅎ 

* 아직 국제 무대에서는 크게 주목받지는 못하지만 그래도 한국어는 화자 수로만 봐도 세계 15위권 안에 들고 진짜로 몇 안되는 featural writing system (갓종갓왕님 1인 프로젝트 덕분에 sign writing이나 소설을 위해 만들어진 언어 등과 같은 문자형식 지위를 획득 ...) 이며 promissive와 같은 독특한 particle 덕분에 언어학 서적에서도 왕왕 나오는 언어 및 문자체계를 갖고 있습니다. 또한 BTS의 떡상으로 전세계 사람들이 한국과 한국어와 한글을 알아서 Korean NLP가 국제무대에서도 Chinese나 Arabic만큼 중요하게 다뤄지는 때가 오기를... 혹은 통일을 기원하며 떡상을 기다리는... 

* 여튼 여기서 한 얘기는 한국어 NLP에서 semantic하게 의미있는 작업을 위해서는 agglutinate된 block들을 morpheme 단위로 적절히 쪼개는 과정이 필수적이라는 얘기를 하고 있습니다. 저는 KoNLPy에 들어 있는 Twitter가 가볍고 자소분해를 하지 않아 많이 사용하는데, 여기서도 해당 모듈을 사용하도록 하겠습니다.

---

The most important part of Korean NLP lies in using Python 3.x. For the lower version, the encoding issue will bother you. That is, in Python 3.x, you don't have to perform an additional encoding for HANGUL to be read and written. [THIS](https://github.com/warnikchow/3i4k/blob/master/data/fci.txt) is the dataset URL, which is a single *.txt* file that has the first column made up of the labels and the second column of the sentences. The dataset is split into train and test set of ratio 9:1 and the test set denotes the last 10% of the corpus. The following code reads the file and makes it into sentences and labels, where the dataset is placed in the path 'data/fci.txt' for the directory you're running the console.

```python
def read_data(filename):
    with open(filename, 'r') as f:
    # if in Window environment, use: 
    # with open(filename, 'r',  encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data
    
fci = read_data('data/fci.txt')
fci_data = [t[1] for t in fci]
fci_label= [int(t[0]) for t in fci]
```

* 한국어엔 역시 Python 3.x 죠.... 2.x로 한국어 인코딩은 지옥입니다 ㅠㅠ 처음에 아주 고생했네요. 어쨌든 제일 첫 삽은 [data](https://github.com/warnikchow/3i4k/blob/master/data/fci.txt)를 로딩하고 sentence와 label들로 나누는 과정입니다. 저는 [Lucy Park님의 유명한 슬라이드](https://www.lucypark.kr/docs/2015-pyconkr.pdf)에서 처음 봤지만, 외국 NLP 튜토리얼들에도 비슷하게 나와 있습니다. 일단 data라는 폴더를 만들어서 *fci.txt* 파일을 넣어주셔야 하며, read_data로 파일을 읽어들이고 탭(\t)으로 split한 후, data와 label array로 나누게 됩니다.

* Daewon Yoon님이 제기해주신 issue를 반영하여, 윈도우 환경에서 생길 수 있는 인코딩 에러를 보완하기 위해 Window environment용 코드를 병기해 두었습니다. 혹시 우분투나 리눅스가 아닌 환경에서 작업하실 경우 해당 코드를 반영해 주시면 될 것 같습니다.

---

The last part of data preprocessing is tokenizing the sentences into morphemes, as emphasized previously. Although many character-level (morpho-syllabic blocks) or sub-character-level (consonants and vowels, or *Jamo*) approaches are utilized these days, the morpheme-level approach is still meaningful due to the nature of Korean as an agglutinative language. For the sparse vector classification such as one-hot encoding and TF-IDF which will be displayed in the following chapter, we will adopt the morpheme sequence which can be obtained by the Twitter tokenizer. For a modified usage of Twitter() class since Konlpy > 0.5, refer to [this documentation](http://konlpy.org/en/latest/api/konlpy.tag/#okt-class).

```python
import numpy as np
import nltk
from konlpy.tag import Twitter
pos_tagger = Twitter()

def twit_token(doc):
    x = [t[0] for t in pos_tagger.pos(doc)]
    return ' '.join(x)

len_train = int(np.floor(len(fci_data)*0.9))
fci_data_train  = fci_data[:len_train]
fci_data_test   = fci_data[len_train:]
fci_label_train = fci_label[:len_train]
fci_label_test  = fci_label[len_train:]

fci_token_train = [twit_token(row) for row in fci_data_train]
fci_token_test  = [twit_token(row) for row in fci_data_test]

fci_sp_token_train = [nltk.word_tokenize(row) for row in fci_token_train]
fci_sp_token_test  = [nltk.word_tokenize(row) for row in fci_token_test]
```

* 데이터를 읽어들였으니, 이제 형태소 분리를 통해 어절이라는 큰 뭉텅이들을 좀 더 세밀한 단위로 나눠 보도록 하겠습니다. 음절 기반 방법, 혹은 자소 분해 방법들도 요즘 많이 사용되지만, 형태소 기반의 문장 표현이 아무래도 교착어인 한국어에서 가장 기본적인 접근 방법이 아닌가 싶습니다. 

* 아까 말씀드렸듯 트위터 형태소 분석기를 사용하여 어절들을 분리할 계획이며, 이는 one-hot encoding이나 TF-IDF를 이용한 sparse vector classification에 활용하기 위함입니다. 위의 코드에서 *twit_token* 함수는 각 문장을 형태소 레벨로 나누어, 형태소 사이에 space를 준 morpheme sequece를 출력합니다. 아래의 코드는 *fci_data*를 train:test ratio 9:1로 split합니다. 

* 위 코드는 Konlpy 0.4.4를 이용한 코드입니다. Konlpy > 0.5 부터 변형된 Twitter() 클래스의 사용을 위해서는 [이 문서](http://konlpy.org/en/latest/api/konlpy.tag/#okt-class)를 참고하세요.

## 2. One-hot encoding and sentence vector
The fundamentals of computational linguistics lies in making machines understand human utterances (or, natural language). Since it is difficult for even human beings to understand the **real meaning**, the first step is to represent the words and sentences into the numerics that are computable for the machine learning systems. Given the dictionary of vocabulary size N, the most famous and intuitive approach is **one-hot encoding** that assigns 1 for only one entry if a specific word is given. Here, the words denote morphemes which are yielded by a morphological analyzer, not the *eojeol* that is the unit of spacing.

* 전산 언어학은 기본적으로 컴퓨터한테 사람이 말하는 것, 즉 자연어를 이해시키는 과정이라고 할 수 있겠습니다. 그래서 우리는 컴퓨터로 하여금, '아 진짜 우리 말을 이해하진 못해도, 대충 어떤 내용인지 판단은 할 수 있게 해 보자' 생각을 하게 됩니다. 그렇게 해서 나오게 된 표현법 (representation) 중 가장 기본적으로 사용되고, 매우 강력한 편이라 아직까지도 많은 곳에서 사용하고 있는 방법론은 바로 단어의 one-hot encoding입니다. 

* 이 방법을 쓰기 위해선 '우리가 생각할 단어 전체 set = *Dict*'을 생각해야 합니다. 가장 먼저 생각해볼 수 있는 건 우리가 사용할 코퍼스에 있는 모든 단어들의 모임이죠. 코퍼스의 사이즈가 커질수록 Dict도 커질 것이고, 물론 단어는 countably infinite하게 만들어낼 수 있지만 여기선 '그나마 자주 쓰이는 녀석들'로 생각합시다. 

* 이 *Dict*가 생각보다 커서 실제로는 보통 많이 쓰이는 n만 단어 같이 제한을 둬서 잡습니다. 목적에 따라 functional한 녀석들은 아예 카운트하지 않기도 해요. 이 때 size(*Dict*) = V 이라 하면, 우리는 (고려하기로 한) 모든 단어들을 V-dim의 one-hot vector로 표현할 수 있게 되는 겁니다. 물론 이전 chapter에서의 결과를 생각한다면, 여기서 word는 morpheme이 되겠죠. 물론 한국어에서 띄어쓰기의 단위인 어절이 영어에서의 word와 조금 더 잘 대응이 되고, 실제로 어절 기반의 processing이 더 정확하게 반영할 수 있는 정보들도 있겠지만, 몇만 개나 되는 morpheme들의 조합이 훨씬 많은 어절 variation을 내포하고 있을 것을 고려하면, 어절을 사용한 one-hot encoding이 얼마나 많은 computation을 요구하게 될지 짐작할 수 있습니다.

---

What should we do with these large-dimensional vectors? The first thing we can think of is a feature called **Bag-of-Words** (BoW). The literal meaning is a bag which contains words, and for Korean, it might be either words (*eojeol*), morphemes, characters, or sub-characters (*Jamo*). Though we've decided only to use the morphemes for sparse representation, these can be replaced with whatever feature you want to adopt. BoW approach is straightforward; we just assign 1 to the corresponding entry if a word in the sentence, and 0 for the others. This can provide the computer with a numerical value which represents the sentence! 

* 어쨌든 우리는 가장 간단한, sparse한 word vector을 만들었습니다. 이걸 갖고 뭘 하느냐? 가장 먼저 생각해볼 수 있는 것은 bag-of-words란 녀석입니다. 말 그대로 '단어가 든 가방' 이에요. 이 때 가방 = 문장 입니다. 문장 안에 어떤 단어들이 들었냐를 V-dim one-hot vector들의 OR-sum operation (하나만 있어도 1됨) 으로 표현하는 거죠. 

* 예컨대 "신이 그댈 사랑해"라는 문장이 있고 '신' = \[1 0 0 0 0 0 0 0\], '이' = \[0 0 1 0 0 0 0 0\], '그댈' = \[0 0 0 1 0 0 0 0\], '사랑' = \[0 0 0 0 1 0 0 0\], '해' = \[0 0 0 0 0 0 1 0\] 로 표현된다고 합시다. 여기서 코퍼스는 여덟 개의 토큰 (단어구성단위 라고 합시다 일단)으로 구성된 아주 작은 *Dict*를 yield했겠지요. 그렇다면 상기 문장은 \[1 0 1 1 1 0 1 0\]의 size(Dict)-dim binary vector로 표현되는 겁니다. 이렇게 해서 뭘 할 수 있냐구요? 이제 컴퓨터도 알아먹는 수치적 정보가 되었으니, 각종 분류기에 넣어 재미를 볼수 있죠! 

* 물론 태클이 들어올 수 있습니다. 저 문장을 사실 '신' '이' 그대' '-ㄹ' '사랑' 'ㅎ' '-애' 로 나눠야 합당하지 않느냐, 한 문장에서 여러 번 카운트되는 단어들이 있으면 one-hot은 부당한 representation이 아니냐 뭐 그런... 첫 번째의 경우 우리가 문장을 자소로 분리하지 않는 Twitter analyzer을 썼기 때문에 어쩔 수 없는 부분입니다. 두 번째의 경우 다음 chapter에서 더 다뤄 보도록 하겠습니다.

## 3. TF-IDF and basic classifiers
Previously, we've introduced one-hot encoding of the words and the sparse sentence representation based on the BoW model. However, despite its transparency and conciseness, one-hot encoding does not convey the word frequency regarding the document. This is where the concept of **term frequency** (TF) came out; the word frequency is taken into account to convey the relative importance of each word. For instance, the word 'I' and 'you' in the sentence *I love you, I want you, I need you* may be assigned the word frequency of 3 instead of 1 which is assigned to the verbs.

<p align="center">
    <image src="https://github.com/warnikchow/dlk2nlp/blob/master/image/tfidf.png" width="400"><br/>
    (Image from https://skymind.ai/wiki/bagofwords-tf-idf)

* 앞서 bag-of-words 모델을 통해 문장을 1-hot vector들의 합(여기서 합은 Boolean의 or에 해당)으로 나타내어 컴퓨터가 알아먹을 만한 어떤 수치로 나타내는 과정을 설명했습니다. 이는 전통적이고 직관적이며 상당히 강력한 방법이기도 합니다.

* 하지만 이 방법의 문제는 문장 내에 등장하는 모든 단어들을, 등장 횟수에 상관없이 공평하게 대한다는 것입니다. 예컨대 "나는 아브라함의 하나님이요 이삭의 하나님이요 야곱의 하나님이로라 하신 것을 읽어 보지 못하였느냐 (전 종교가 없습니다! 그냥 예문을 찾고싶었을뿐)" 라는 문장이 하고 싶은 말은 누가 봐도 내가 하나님이란 것 같은데 아브라함이니 이삭이니 하는 것들과 동급으로 1로 카운트되면 얼마나 억울하겠습니까. 

* 이러한 문제를 해결하기 위해 우린 일종의 normalization으로 볼 수 있는 term frequency, 즉 문장의 전체 word중 해당 word의 빈도를 고려해 문장을 벡터화해 볼 수 있습니다. 위의 문장에선 '하나님'이 3/X (형태소분석한 결과를 모두 카운트하기 귀찮으니 X로 퉁칩시다)의 값을 부여받고 나머지 녀석들은 1/X의 값을 부여받게 되겠죠.

---

However, the problem is that the high frequency does not indicate the importance of the word. Thus here, the concept of **inverse-document frequency** (IDF) is introduced, as a way of multiplying the inverse fraction of the frequency of the word among the whole documents. This prevents the overestimation of the functional words in many cases; to be honest, 'I' and 'you' are not as important as 'love', 'want' and 'need' in most cases (unless if the task deals with directivity). For instance, in the morpheme-level analysis, many particles in Korean are used repetitively in the sentences; those will be assigned a low IDF so that the lexical words are emphasized alternatively. The following code utilizing Scikit-learn library demonstrates how the **term frequency-inverse document frequency** (TF-IDF) is computed for our corpus.

```python
# Referred to the followings for the code:
# https://gist.github.com/jason-riddle/1a854af26562c0cdb1e6ff550d1bf32d#file-complex-tf-idf-example-py-L40
# http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_featurizer(corpus_train,corpus_test):
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(corpus_train)
    freq_term_matrix = count_vectorizer.transform(corpus_train)
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)
    # unigram features
    tfidf_token = TfidfVectorizer(ngram_range=(1,1),max_features=3000)
    token_tfidf_total = tfidf_token.fit_transform(corpus_train+corpus_test)
    token_tfidf_total_mat = token_tfidf_total.toarray()
    # bigram features
    tfidf_bi_token = TfidfVectorizer(ngram_range=(1,2),max_features=3000)
    token_tfidf_bi_total = tfidf_bi_token.fit_transform(corpus_train+corpus_test)
    token_tfidf_bi_total_mat = token_tfidf_bi_total.toarray()
    return token_tfidf_total_mat,token_tfidf_bi_total_mat

fci_tfidf,fci_tfidf_bi  = tfidf_featurizer(fci_token_train,fci_token_test)
fci_tfidf_train    = fci_tfidf[:len_train]
fci_tfidf_test     = fci_tfidf[len_train:]
fci_tfidf_bi_train = fci_tfidf_bi[:len_train]
fci_tfidf_bi_test  = fci_tfidf_bi[len_train:]
```

* 그런데 또 문제가 있습니다. 바로 하나님이 소유를 나타내는 '의'와 동급이 돼버리는 겁니다. 아아, 이렇게 원통할 데가... 딱 봐도 '의'라는 녀석은 문장의 의미를 판단하는 데에 큰 도움을 주지 않을 것으로 보입니다. 다른 문장들에도 많이 나올 게 분명하거든요. 

* 그래서 우리가 생각해볼 수 있는 건 '다른 문장들에도 많이 나오는 녀석엔 가중치를 조금 주면 어떨까?'하는 것입니다. 예컨대 전체 문장 중 해당 문장이 나오는 비율의 역수 같은 걸 곱해준다면? 이런 생각으로 나온 녀석이 바로 inverse document frequency입니다. 약간의 smoothing factor을 추가하자면, test corpus에 해당 term이 없는 경우를 대비해 분모에 1을 더해주고, corpus size가 방대해질 때를 고려해 log를 입히는 정도?

* 상기한 두 개의 요소를 곱해 BoW를 개량한 모델이 바로 TF-IDF (term frequency-inverse document frequency) 입니다. 문서 분류에 아직도 활발히 사용되는 모델이지요. 위의 코드에서 *tfidf_vectorizer* 함수는 train corpus에서 tf-idf statistics를 뽑아내고 이를 이용해 train corpus+test corpus를 수치화하는 과정을 담고 있습니다. statistics를 뽑아낼 때 train corpus만 사용하는 것은, test corpus까지 되면 prediction의 본래 의미에 맞지 않게 되기 때문이죠. 수치화하는 과정은 이 statistics를 이용하여 *fit_transform*의 함수를 통해 일괄적으로 진행됩니다.

---
The aforementioned sentence representation can be directly utilized with basic classifiers such as naive Bayes (NB), decision tree (DT), support vector machine (SVM) and logistic regression (LR). The evaluation is done with accuracy and F1-score; accuracy refers to the fraction of correct instances out of the whole data. Understanding the meaning of the value is intuitive, but the flaw is that it does not convey how incorrect the model predicts for the classes of data with a small portion. For example, if the data consists of the binary label and the portion is 9:1 yielding an imbalance, then the accuracy may reach 90% just by a simple guess of a class with the larger volume. A better solution can be obtained by using F1-score, which considers the true negatives and the false positives. The following code displays how the sparse vectors we previously obtained are used in training and prediction, with both an accuracy and an F1-score.

```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

classifier_uni = LogisticRegression(random_state=1234)
classifier_uni.fit(fci_tfidf_train,fci_label_train)
uni_pred = classifier_uni.predict(fci_tfidf_test)
accuracy_score(uni_pred,fci_label_test)
metrics.f1_score(uni_pred,fci_label_test,average="macro")
metrics.f1_score(uni_pred,fci_label_test,average="weighted")
precision_recall_fscore_support(uni_pred,fci_label_test)

classifier_bi = LogisticRegression(random_state=1234)
classifier_bi.fit(fci_tfidf_bi_train,fci_label_train)
bi_pred = classifier_bi.predict(fci_tfidf_test)
accuracy_score(bi_pred,fci_label_test)
metrics.f1_score(bi_pred,fci_label_test,average="macro")
metrics.f1_score(bi_pred,fci_label_test,average="weighted")
precision_recall_fscore_support(bi_pred,fci_label_test)
```

* 지금까지  tf-idf를 이용한 sentence 의 sparse representation에 대해 얘기했지요. 이제 구체적으로 이 녀석들을 문장 분류에 이용해먹을 만한 방법들에 대해 생각해봐야 하는데요, 그 중 하나가 이 글의 task로 제시된 분류 (classification)입니다. 

* 사실 데이터 집합(문장들)과 레이블 집합(긍정문/부정문, 의문문별 분류, 토픽별 분류 등)으로 된 트레이닝 데이터를 input으로 넣어, 별도의 테스트 데이터로 얻어진 prediction와 정답의 비교를 통해 그 네트워크의 accuracy, F-measure 등을 evaluate하여 성능을 평가한다는 점은 많은 분야에서 사용되는 분류기 evaluation의 구조입니다. 

* 이 때 accuracy는 전체 테스트 인풋 중 분류에 성공한 input의 비율이 될 겁니다. 그런데 함정이 있다면, 데이터 편중을 배제하기 어렵단 것이죠. 예컨대 데이터셋의 90퍼센트가 긍정문이고 10퍼센트만 부정문이라면, 공정한 트레이닝 및 테스트가 되었다고 하기 어렵겠죠. 그래서 binary classification에서 많이 사용되는 F-measure의 경우는 precision과 recall이라는, 각각 '1이라고 했을 때 정말 맞을 확률'과 '전체 맞은 것들 중 1이라고 했을 확률'을 생각하여 이들의 조화평균을 구해주는 방향으로 accuracy의 함정을 보정합니다. 물론 단순 조화평균이 아닌 별도 parameter을 주기도 하고, multiclass에 대해선 따로 class별 weight를 정의하기도 하지만요.

---

The result is as below; it is weird that the bigram features yielded quite a lower accuracy. It is assumed that the property of the corpus, where the sentence label does not depend solely on the polarity items or emotion words, may have affected the performance. Also, the dimension of the feature (3,000) which was fixed to guarantee the fair comparison between the two models, could have been a harsh obstacle for the bigram model.

```properties
# CONSOLE RESULT
>>> accuracy_score(uni_pred,fci_label_test)
0.7742409402546523
>>> metrics.f1_score(uni_pred,fci_label_test,average="macro")
0.6000194409152938
>>> metrics.f1_score(uni_pred,fci_label_test,average="weighted")
0.7880295802371478
>>> precision_recall_fscore_support(uni_pred,fci_label_test)
(array([0.74875208, 0.87206124, 0.88099174, 0.73828125, 0.14534884,
       0.23148148, 0.32398754]), array([0.70754717, 0.71046771, 0.84513742, 0.82749562, 0.5952381 ,
       0.80645161, 0.75362319]), array([0.72756669, 0.78301424, 0.8626922 , 0.78034682, 0.23364486,
       0.35971223, 0.45315904]), array([ 636, 2245, 1892, 1142,   42,   31,  138]))
       
>>> accuracy_score(bi_pred,fci_label_test)
0.321743388834476
>>> metrics.f1_score(bi_pred,fci_label_test,average="macro")
0.18947002986641723
>>> metrics.f1_score(bi_pred,fci_label_test,average="weighted")
0.35601345547363006
>>> precision_recall_fscore_support(bi_pred,fci_label_test)
(array([0.82529118, 0.46473483, 0.18787879, 0.22109375, 0.        ,
       0.        , 0.00311526]), array([0.28101983, 0.36340316, 0.34236948, 0.28936605, 0.        ,
       0.        , 0.05      ]), array([0.41927303, 0.40786948, 0.24261829, 0.2506643 , 0.        ,
       0.        , 0.0058651 ]), array([1765, 2339,  996,  978,   12,   16,   20]))
```

* 결과는 뜻밖이었는데요, bigram으로 구한 결과가 그다지 신통치 않은 것 같습니다. (코드를 잘못 짰나...?) 어쨌든 궁색하게 분석을 해보자면, 본 task가 sentiment analysis 처럼 단어 한두개로 문장의 의미가 크게 달라지는 task들과 다르게 전체적인 맥락 및 어순을 고려해야 하기도 하고, fair comparison을 위해 3,000으로 제한한 feature dimension이 오히려 bigram에는 해가 되어 정작 중요한 형태소들을 놓친 게 아닌가...하는 생각이 듭니다.

* 앞서 구한 one-hot encoding으로도 이러한 evaluation을 손쉽게 할 수 있습니다. 방금 TF-IDF에서 그랬듯, evaluation을 위한 prediction은 10%의 test data로 하게 되지요. Bigram TF-IDF는 좀 불만족스럽지만, 어쨌든 sparse representation으로도 어느 정도는 높은 정확도를 가진 모델을 얻을 수 있음을 알 수 있습니다. 물론 일부 class들에 대해선 random guess만도 못한 결과를 내고 있지만요. 다만 아직도 찝찝한 점은, 컴퓨터가 단어를 세고만 있지 그 단어들이 문장 내에서, 작게는 컨텍스트에서 어떤 역할을 하는지 아무것도 모르는 것 같다는 점입니다. 

## 4. Dense word embeddings

Three-line summary:</br>
1. Computational linguistics aims making machines understand human language.</br>
2. As a fundamental approach for the representation of words and sentences, one-hot encoding and TF-IDF are introduced.</br>
3. For the Korean language, due to the property of the agglutinative language, morpheme-level analysis can be more effective than the word (*eojeol*)-level one.</br>

However, considering the computation issue which has been crucial up to this date, the sparse representations may not be the optimal solution for the contemporary neural network-based systems. This is the point where the dense representation such as [*Word2Vec*](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) is successfully adopted.

* 여태까지 한 내용을 세줄요약해 보면 다음과 같습니다.</br>
1. 전산언어학은 기계로 하여금 사람 말을 잘 알아먹도록 하는 학문</br>
2. 단어 및 문장을 모델링하는 기본적인 방법으로 one-hot encoding (BoW) 및 TF-IDF가 있음</br>
3. 한국어는 교착어적 특징 때문에 어절 단위 분석보다 형태소 단위 분석이 효과적일 수 있음

* 이외에도 수치화된 문장을 분류하는 알고리즘에는 SVM 이나 LR 등이 있다, 그 성능을 평가하는 measure에는 accuracy나 F1 score 등이 있다 ... 라는 것도 간략히 말하고 지나왔죠. 

* 그런데 이러한 방법론들은 분석해야 할 데이터가 많아지고 사전의 크기가 커질수록 computation의 문제에 당면하게 됩니다. 특히나 데이터를 몰빵해넣고 병렬연산으로 승부하는 딥러닝의 경우 더욱 그렇지요. 예컨대 30 개의 형태소로 된 문장을 벡터 sequence로 나타내어 recurrent neural network 같은 시스템으로 요약하고자 할 때, 딕셔너리 사이즈가 10만이라면, 10만 x 30이라는 무시무시한 사이즈의 행렬이 문장 하나를 표현하여 인풋으로 들어가게 되는거죠. 물론 아까 보았듯 사용하는 벡터의 크기는 조절할 수 있고, 각 벡터가 각 단어를 표현하는 것이 아주 명확하긴 하지만요.

* 이럴 때 유용하게 사용되는 개념이 2013년 태동한 word2vec, 혹은 dense low-dimension embedding 입니다. 등장 목적이 위와 같다고는 할 수 없지만, [2006년부터 촉발된 딥러닝 아키텍쳐](http://www.cs.toronto.edu/~fritz/absps/ncfast.pdf)와 맞물려 사용되며 문장 분석의 패러다임 자체를 바꾸고 있죠. Word2vec, GloVe, fastText에 대한 설명을 간략하게만이라도 해보려 합니다.

---

The term word2Vec is very intuitive, and it converts the words, which are discrete (and were sparsely represented so far), into the numerics that are close to continuousness. Since there are a [bunch of](https://skymind.ai/wiki/word2vec) [terrific](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa) [articles](https://www.tensorflow.org/tutorials/representation/word2vec) on the topic outside, in English, a review on word2vec and its related models are discussed mainly in Korean.

<p align="center">
    <image src="https://github.com/warnikchow/dlk2nlp/blob/master/image/skipgram.png" width="400"><br/>
    (Image from [the tutorial site](https://sheffieldnlp.github.io/com4513-6513/lectures_reveal_js/lecture12_word_embeddings.html). The matrix W x D is obtained by minimizing the cost function regarding the performance of context prediction given an input vector.)

* word2vec이라는 말은 상당히 직관적입니다. 말 그대로 이산적 개념인 단어를 수치적 개념인 벡터로 바꿔 준다는 의미이죠. 사실 one-hot encoding 역시 고차원의 벡터를 만들어준다는 점을 생각하면 어폐가 있긴 합니다. 그래서 두 개념의 차이를 벡터가 sparse한지 (드문드문하게 nonzero인 성분이 있는지), 아니면 dense한지 (유클리드 공간에서처럼 빽빽이 들어차 있는지)를 차이점으로 봅니다. 물론 word2vec은 후자를 의미하죠.

* 당연한 얘기겠지만, 어떤 방식의 워드 임베딩이든, 임베딩 벡터 셋을 트레이닝하는 과정에서 어떤 원칙 (혹은 제약조건)을 주냐에 따라 결과물로 나오는 벡터들 간의 관계가 달라지게 됩니다. 예컨대 아무 제약 조건도 주지 않는 one-hot encoding의 경우, 모든 단어가 평등합니다. 아무리 비슷해 보이는 단어들이라도 1이 위치하는 엔트리가 다르다면 아무 상관없는 단어인 것이나 다름없게 되는 것이죠. distributional semantics에 대한 고려는 one-hot vector에 들어 있지 않은 겁니다.

* 모든 sparse vector가 one-hot vector같이 엔트리 간 equivalence를 지니는 건 아닙니다. 오히려 희소한 케이스에 가깝죠. 가중치를 곱해준 TF-IDF만 해도, 개별 단어를 모델링하지는 않지만 어떤 분포적인 특성이 반영되게 됩니다. binary이지만 multi-hot encoding을 차용하는 [boolean distributional semantics](https://transacl.org/ojs/index.php/tacl/article/view/616)의 경우, 의미론에서 얘기하는 feature의 개념이 등장하여, taxonomy 상 상위 위계에 해당하는 단어의 embedding이 하위 위계의 단어의 embedding에 포함되게 하는 방식으로 트레이닝이 됩니다. 아마도 벡터 사이즈는 one-hot보다는 줄어들겠죠? 벡터들 간의 distance measure가 euclidean이 아닌 어떤 이산적인 개념이 된다는 건 challenging하긴 합니다만, 언어가 기본적으로 이산적인 속성을 버릴 수 없다는 생각을 갖고 있는 저로썬 매우 매력적이라는 생각을 했습니다.

* 그렇지만 이상은 이상이고, 우리는 많은 순간 현실과 타협해야 합니다. 작은 벡터에 정보들을 우겨 넣어야 하죠. 또한 back propagation과 같은 연산을 통해 이루어지는 최신 최적화 기법들의 수혜를 받으려면, 벡터 간의 거리가 어떤 미분가능한, 이산적이지 않은 개념으로 정의가 되어야 함도 사실입니다 (물론 이산적인 목적 함수들을 위해 별도의 신경망을 연결해 주는 알고리즘도 있는 것으로 압니다만 일단 그건 나중에 생각하도록 하지요). 그러기 위해서, 고차원의 벡터를 저차원에 임베딩해 넣으면서도, 단어들 간의 관계가 좀 더 유기적으로 연결될 수 있는 방법엔 뭐가 있을까요? 

* 이러한 관점에서 볼 때, word2vec은 상당히 매력적인 시도였다 볼 수 있겠습니다. 혹자는 '비슷한 context에 등장하는 녀석들은 실제로도 비슷한/관련 있는 녀석들일 가능성이 높다는 distributional semantics의 원칙을 수치적 제약으로 잘 지키면서 one-hot vector을 저차원에 pca한 결과물'이라고 하더군요. 예컨대 '너는 나쁜 아이야'와 '너는 착한 아이야'라는 문장들을 볼 때, '나쁜'과 '착한'이 실제로 저런 류의 context를 많이 공유한다 생각해 보면, 둘이 아주 관계가 없는 단어들은 아니다, 어느정도 유기적이다, 그런 판단을 할 수 있겠죠? word2vec의 큰 철학은 이렇습니다. 컨텍스트를 주고 center word를 추론하는 CBOW와 그 반대인 skip-gram (SG) 모두 word2vec의 최적화와 관련된 알고리듬인데요, SG가 여러 태스크에서 더 성능이 좋음이 보여진 바 있고 실제로도 더 자주 활용됩니다. 

---

Up to date, many advanced models of word vectors which base on word2vec have been proposed. For instance, [**GloVe**](https://nlp.stanford.edu/pubs/glove.pdf) represents the word vectors considering the co-occurrence in the window, and [**fastText**](https://arxiv.org/abs/1607.04606) utilizes subword n-gram so that the embedding can be efficient for morphologically rich languages. In the following approaches that use dense word vectors, we adopt [100-dimension fastText vector dictionary which was trained with 2M drama scripts](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor), for the project [RAWS](github.com/warnikchow/raws). Use Gensim if fastText is not installed.

```python
import fasttext
model_ft = fasttext.load_model('vectors/model_drama.bin')

# In case fasttext is not installed
# use FastText wrapper in Gensim as:
# from gensim.models.wrappers import FastText
# model_ft = FastText.load_fasttext_format('vectors/model_drama.bin')
```

* Word2vec은 이런 저차원 임베딩의 포문을 열었을 뿐이고 (for문 아닙니다) 이후로 수많은 베리에이션이 등장했습니다. 이 때 사용하는 알고리듬이 skip-gram이란 점이 크게 변하지는 않았지만, 다양한 objective를 설정하며 특정 task에 효율적으로 적용할 수 있는 vector들이 등장했죠.

* 가장 많이 알려진 두 베리에이션은 GloVe와 fastText입니다. 전자는 word2vec 이전에 사용되던 sparse representation 중 하나인 co-occurrence matrix의 개념을 training objective에 활용하여 보다 global하고 syntactic한 특징도 word vector에 반영할 수 있게 한 것이고, 후자는 word2vec과 사실상 같은 알고리즘이지만 그 트레이닝 효율을 거의 몇백배 수준으로 끌어올려 빠르게 트레이닝하면서도 task 성능은 엇비슷하게 유지할 수 있게 하는 알고리듬입니다. 또한 fastText는 subword n-gram model을 차용하여, word 단위로 나뉘어지지 않은, 단어 내의 character n-gram들도 일종의 word로 보고 그 분포를 전체 트레이닝에 고려한다는 특징을 가지고 있지요. 이를 통해 morphologically rich한 언어들에서도 효율적으로 활용 가능하다는 점을 논문에서 어필하고 있구요.

* GloVe의 경우 stanford에서 제공하는 [wiki/twitter기반 pre-trained vector](https://nlp.stanford.edu/projects/glove/)가 있으며, fastText의 경우는 꽤 많은 언어로 pre-trained vector을 제공하지만 학습 속도가 굉장히 빨라 저 같은 경우는 갖고 있는 코퍼스로 새로 training하기도 합니다. Subword n-gram이 교착어인 한국어에도 굉장히 유용하여, 저 같은 경우는 fastText로 트레이닝된 word vector set을 character embedding에 사용하고 있구요. 약 200만 문장의 드라마 스크립트를 통해 학습한 word vector dictionary는 [다음의 주소](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor)에서 제공됩니다. [fasttext 라이브러리](https://pypi.org/project/fasttext/0.8.3/)를 이용해 bin 파일을 바로 load할 수 있으며, 설치가 잘 되지 않을 경우 Gensim의 FastText wrapper을 이용해 비슷한 방식으로 load가 가능합니다.

* 앞서도 말했지만, 26개의 letter로 된 알파벳만 있으면 되는 영어와 달리, 한국어는 2500개 상당의 자모 조합이 있어 one-hot encoding을 직접적으로 character-level embedding에 사용하기 쉽지 않죠. 이럴 때 유용하게 사용할 수 있는 것이 저차원으로 임베딩된 character vector입니다. 어느 정도 분포적인 특징을 반영할 수 있으면서도 computational하게 부담을 덜 줄 수 있는 그런 feature로 사용할 수 있는 것입니다. 물론 형태소, 어절 모두 임베딩의 대상이 될 수 있습니다. 어떤 것을 선택할지는 형태소 분석기의 유무, 구동 및 개발 환경 등에 따라 자유롭게 선택하면 될 것입니다. 

## 5. Document vectors and NN classifier

In the first few chapters, we've demonstrated how the corpus we adopted is preprocessed, featurized, trained and used in prediction with the sparse sentence encodings. However, since we've obtained the dense word embeddings for the morphemes (as we've obtained for one-hot encoded words), it's plausible to extend it to the sentence vector, for instance by summation. 

```python
from numpy import linalg as la

def featurize_nn(corpus,wdim):
    nn_total = np.zeros((len(corpus),wdim))
    for i in range(len(corpus)):
        if i%1000 ==0:
            print(i)
        s = corpus[i]
        count_word=0
        for j in range(len(s)):
            if s[j] in model_ft:
                nn_total[i,:] = nn_total[i,:] + model_ft[s[j]]
                count_word = count_word + 1
        if count_word > 0:
            nn_total[i] = nn_total[i]/la.norm(nn_total[i])
    return nn_total

fci_nn = featurize_nn(fci_sp_token_train+fci_sp_token_test,100)
```

* 앞서 one-hot encoding과 tf-idf로 문장을 수치화하는 방법, 그리고 그것이 sparse vector이라는 것까지 말씀드린 바 있습니다. 정보가 discrete하다는 것은 기계에게나 사람에게나 매우 직관적으로 다가오기에, 문장의 분류에 있어 굉장히 유용하게 사용된다는 것두요. 하지만 지난 몇 시간 동안 dense word vector들을 얘기하는 동안, 그 친구들을 문장 수치화에 이용할 수 있지 않을까 생각하는 분도 분명 계실 겁니다. word2vec을 제안한 사람들의 머릿속을 들여다볼 수는 없지만, 결과적으로 그 등장이 문장 임베딩에 있어 하나의 패러다임 전환이 되었죠.  

* “I really love you” 이라는 영어 문장을 한번 생각해 봅시다. 일단 가장 먼저 생각해볼 수 있는 것은, \[0 0 0 ... 1 ... 1 ... 1 ... 1 ... 0 0 0\] 이런 식으로 나타내어진 multi-hot vector (one-hot vectors의 or summation)이겠죠. 그 다음은 \[0 0 0 ... 0.3 ... 0.7 ... 0.9 ... 0.4 ... 0 0 0\] 이런 식으로 표현된 tf-idf일 겁니다. 이제 word2vec이 무엇인지 배웠으므로, 이런 방법도 생각해 볼 수 있을 겁니다. ‘문장 내 모든 word와 word embedding function f에 대해, f(w)를 모두 더하기’ 즉, 워드 임베딩이 100차원의 real vector이라면, 전체 sentence vector s=sum(f(w))도 100차원의 real vector가 되는 방법이죠. 여기에 normalization을 위해 l2_norm(s)나 word 개수 (여기서는 4)로 s를 나눠 주면 좀 더 reliable한 표현이 될 것입니다. 앞서 말한 과정이 *featurize_nn*에 구현되어 있으며, 해당 함수는 corpus 전체를 input으로 받아 각 문장마다 100차원 (word vector dim)의 real vector을 출력합니다.

---

And here comes Keras, which is a widely used high-level wrapper for TensorFlow (and other libraries i.e. Theano and CNTK; though not used in general). Since we only use *Sequential()* for the codes not incorporating the concatenated layers or attention models, only *layer* will be imported. For the computation of average/weighted F1 score per epoch, an additional module is defined here. Also, considering the imbalance of the class volumes, we obtain the class weight set that is utilized in the training session.

```python
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))
from keras.models import Sequential
import keras.layers as layers
from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
from keras.callbacks import ModelCheckpoint

from keras.callbacks import Callback
from sklearn import metrics
class Metricsf1macro(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1s_w = []
        self.val_recalls_w = []
        self.val_precisions_w = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_predict = np.argmax(val_predict,axis=1)
        val_targ = self.validation_data[1]
        _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
        _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
        _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
        _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
        _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
        _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_f1s_w.append(_val_f1_w)
        self.val_recalls_w.append(_val_recall_w)
        self.val_precisions_w.append(_val_precision_w)
        print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
        print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro = Metricsf1macro()

from sklearn.utils import class_weight
class_weights_fci = class_weight.compute_class_weight('balanced', np.unique(fci_label), fci_label)
```

* 본격적으로 Keras를 써볼 때가 왔습니다. 일단 제가 TensorFlow를 제대로 써본 적은 없지만, conctenated layer나 attention model같은 복잡한 시스템을 포함하지 않는 대부분의 모델들에 대해서는 *Sequential()* 모듈을 import하는 것으로 대부분 구현 가능합니다 ㅎㅎ TF-IDF의 evaluation에서 활용한 average/weighted F1 score을 epoch마다 계산하기 위해 별도의 함수를 정의했으며, class imbalance를 고려하여 weight set을 구했습니다. 다음의 코드에서 활용되는 것을 볼 수 있습니다.

---

And the Below is the model construction and evaluation phase. Note that the folder *tutorial* was created in the same directory to save checkpoint models, recording F1 scores and the accuracy. It is quite surprising that a simple summation boosts the accuracy and F1 score by a large factor, even considering that the concept of making up the sentence vector is fundamentally identical to that of one-hot encoding and TF-IDF!

```python
def validate_nn(result,y,hidden_dim,cw,filename):
    model = Sequential()
    model.add(layers.Dense(hidden_dim, activation = 'relu', input_dim=len(result[0])))
    model.add(layers.Dense(int(max(y))+1, activation='softmax'))
    model.summary()
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    model.fit(result,y,validation_split=0.1,epochs=30,batch_size=16,callbacks=callbacks_list,class_weight=cw)

validate_nn(fci_nn,fci_label,128,class_weights_fci,'model/tutorial/nn')
```

```properties
# CONSOLE RESULT
>>> validate_nn(fci_nn,fci_label,128,class_weights_fci,'model/tutorial/nn')
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 128)               12928     
_________________________________________________________________
dense_4 (Dense)              (None, 7)                 903       
=================================================================
Total params: 13,831
Trainable params: 13,831
Non-trainable params: 0
_________________________________________________________________
Train on 55129 samples, validate on 6126 samples
Epoch 1/30
54960/55129 [============================>.] - ETA: 0s - loss: 0.8674 - acc: 0.7068— val_f1: 0.591953 — val_precision: 0.721290 — val_recall: 0.549977
— val_f1_w: 0.731234 — val_precision_w: 0.747052 — val_recall_w: 0.742899
55129/55129 [==============================] - 6s 116us/step - loss: 0.8672 - acc: 0.7069 - val_loss: 0.7665 - val_acc: 0.7429
Epoch 2/30
55056/55129 [============================>.] - ETA: 0s - loss: 0.7255 - acc: 0.7537— val_f1: 0.642565 — val_precision: 0.721759 — val_recall: 0.602373
— val_f1_w: 0.754185 — val_precision_w: 0.761563 — val_recall_w: 0.760366
55129/55129 [==============================] - 7s 119us/step - loss: 0.7253 - acc: 0.7537 - val_loss: 0.7111 - val_acc: 0.7604
Epoch 3/30
54992/55129 [============================>.] - ETA: 0s - loss: 0.6856 - acc: 0.7660— val_f1: 0.650076 — val_precision: 0.731066 — val_recall: 0.608824
— val_f1_w: 0.760444 — val_precision_w: 0.769572 — val_recall_w: 0.766569
55129/55129 [==============================] - 7s 119us/step - loss: 0.6856 - acc: 0.7660 - val_loss: 0.6827 - val_acc: 0.7666
Epoch 4/30
54704/55129 [============================>.] - ETA: 0s - loss: 0.6587 - acc: 0.7760— val_f1: 0.665685 — val_precision: 0.750019 — val_recall: 0.625693
— val_f1_w: 0.768111 — val_precision_w: 0.777307 — val_recall_w: 0.773914
55129/55129 [==============================] - 7s 119us/step - loss: 0.6585 - acc: 0.7761 - val_loss: 0.6611 - val_acc: 0.7739
Epoch 5/30
54912/55129 [============================>.] - ETA: 0s - loss: 0.6361 - acc: 0.7839— val_f1: 0.675093 — val_precision: 0.756560 — val_recall: 0.633105
— val_f1_w: 0.772584 — val_precision_w: 0.779329 — val_recall_w: 0.779465
55129/55129 [==============================] - 6s 117us/step - loss: 0.6363 - acc: 0.7838 - val_loss: 0.6531 - val_acc: 0.7795
Epoch 6/30
54752/55129 [============================>.] - ETA: 0s - loss: 0.6175 - acc: 0.7909— val_f1: 0.677678 — val_precision: 0.760474 — val_recall: 0.638087
— val_f1_w: 0.782750 — val_precision_w: 0.789371 — val_recall_w: 0.788116
55129/55129 [==============================] - 6s 116us/step - loss: 0.6178 - acc: 0.7908 - val_loss: 0.6286 - val_acc: 0.7881
Epoch 7/30
55056/55129 [============================>.] - ETA: 0s - loss: 0.6018 - acc: 0.7958— val_f1: 0.698941 — val_precision: 0.753141 — val_recall: 0.667789
— val_f1_w: 0.790828 — val_precision_w: 0.793462 — val_recall_w: 0.795462
55129/55129 [==============================] - 6s 117us/step - loss: 0.6021 - acc: 0.7957 - val_loss: 0.6192 - val_acc: 0.7955
Epoch 8/30
54736/55129 [============================>.] - ETA: 0s - loss: 0.5884 - acc: 0.8009— val_f1: 0.699604 — val_precision: 0.766330 — val_recall: 0.662288
— val_f1_w: 0.794905 — val_precision_w: 0.797012 — val_recall_w: 0.800033
55129/55129 [==============================] - 7s 120us/step - loss: 0.5887 - acc: 0.8008 - val_loss: 0.6067 - val_acc: 0.8000
Epoch 9/30
54880/55129 [============================>.] - ETA: 0s - loss: 0.5753 - acc: 0.8059— val_f1: 0.703015 — val_precision: 0.768262 — val_recall: 0.665053
— val_f1_w: 0.800541 — val_precision_w: 0.807472 — val_recall_w: 0.804114
55129/55129 [==============================] - 7s 118us/step - loss: 0.5758 - acc: 0.8057 - val_loss: 0.5961 - val_acc: 0.8041
Epoch 10/30
55040/55129 [============================>.] - ETA: 0s - loss: 0.5649 - acc: 0.8090— val_f1: 0.707821 — val_precision: 0.784078 — val_recall: 0.666537
— val_f1_w: 0.804109 — val_precision_w: 0.811529 — val_recall_w: 0.809011
55129/55129 [==============================] - 6s 116us/step - loss: 0.5646 - acc: 0.8091 - val_loss: 0.5840 - val_acc: 0.8090
Epoch 11/30
55056/55129 [============================>.] - ETA: 0s - loss: 0.5552 - acc: 0.8127— val_f1: 0.711974 — val_precision: 0.784070 — val_recall: 0.671254
— val_f1_w: 0.809180 — val_precision_w: 0.819130 — val_recall_w: 0.813092
55129/55129 [==============================] - 7s 118us/step - loss: 0.5550 - acc: 0.8128 - val_loss: 0.5850 - val_acc: 0.8131
Epoch 12/30
55008/55129 [============================>.] - ETA: 0s - loss: 0.5454 - acc: 0.8167— val_f1: 0.714970 — val_precision: 0.773000 — val_recall: 0.679629
— val_f1_w: 0.807968 — val_precision_w: 0.811567 — val_recall_w: 0.810970
55129/55129 [==============================] - 7s 119us/step - loss: 0.5459 - acc: 0.8165 - val_loss: 0.5757 - val_acc: 0.8110
Epoch 13/30
54880/55129 [============================>.] - ETA: 0s - loss: 0.5371 - acc: 0.8196— val_f1: 0.708290 — val_precision: 0.802882 — val_recall: 0.662145
— val_f1_w: 0.808888 — val_precision_w: 0.817245 — val_recall_w: 0.814887
55129/55129 [==============================] - 7s 118us/step - loss: 0.5369 - acc: 0.8196 - val_loss: 0.5711 - val_acc: 0.8149
Epoch 14/30
55008/55129 [============================>.] - ETA: 0s - loss: 0.5295 - acc: 0.8216— val_f1: 0.720639 — val_precision: 0.784796 — val_recall: 0.684186
— val_f1_w: 0.813487 — val_precision_w: 0.820760 — val_recall_w: 0.816520
55129/55129 [==============================] - 7s 119us/step - loss: 0.5294 - acc: 0.8216 - val_loss: 0.5735 - val_acc: 0.8165
Epoch 15/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.5218 - acc: 0.8245— val_f1: 0.723226 — val_precision: 0.793214 — val_recall: 0.682388
— val_f1_w: 0.813296 — val_precision_w: 0.821299 — val_recall_w: 0.817009
55129/55129 [==============================] - 7s 118us/step - loss: 0.5217 - acc: 0.8246 - val_loss: 0.5625 - val_acc: 0.8170
Epoch 16/30
54784/55129 [============================>.] - ETA: 0s - loss: 0.5147 - acc: 0.8259— val_f1: 0.718378 — val_precision: 0.805836 — val_recall: 0.671866
— val_f1_w: 0.812241 — val_precision_w: 0.821222 — val_recall_w: 0.817009
55129/55129 [==============================] - 7s 118us/step - loss: 0.5150 - acc: 0.8259 - val_loss: 0.5591 - val_acc: 0.8170
Epoch 17/30
55056/55129 [============================>.] - ETA: 0s - loss: 0.5085 - acc: 0.8286— val_f1: 0.710259 — val_precision: 0.800253 — val_recall: 0.667554
— val_f1_w: 0.811823 — val_precision_w: 0.817403 — val_recall_w: 0.818642
55129/55129 [==============================] - 7s 118us/step - loss: 0.5087 - acc: 0.8285 - val_loss: 0.5605 - val_acc: 0.8186
Epoch 18/30
54704/55129 [============================>.] - ETA: 0s - loss: 0.5026 - acc: 0.8304— val_f1: 0.719479 — val_precision: 0.797672 — val_recall: 0.676889
— val_f1_w: 0.816203 — val_precision_w: 0.823536 — val_recall_w: 0.820764
55129/55129 [==============================] - 7s 119us/step - loss: 0.5029 - acc: 0.8303 - val_loss: 0.5533 - val_acc: 0.8208
Epoch 19/30
54784/55129 [============================>.] - ETA: 0s - loss: 0.4975 - acc: 0.8326— val_f1: 0.718767 — val_precision: 0.790945 — val_recall: 0.677815
— val_f1_w: 0.815264 — val_precision_w: 0.819433 — val_recall_w: 0.820601
55129/55129 [==============================] - 6s 118us/step - loss: 0.4972 - acc: 0.8328 - val_loss: 0.5533 - val_acc: 0.8206
Epoch 20/30
54704/55129 [============================>.] - ETA: 0s - loss: 0.4914 - acc: 0.8341— val_f1: 0.732335 — val_precision: 0.777586 — val_recall: 0.702288
— val_f1_w: 0.817729 — val_precision_w: 0.819907 — val_recall_w: 0.820927
55129/55129 [==============================] - 6s 118us/step - loss: 0.4915 - acc: 0.8340 - val_loss: 0.5515 - val_acc: 0.8209
Epoch 21/30
54832/55129 [============================>.] - ETA: 0s - loss: 0.4868 - acc: 0.8365— val_f1: 0.730587 — val_precision: 0.787088 — val_recall: 0.694430
— val_f1_w: 0.816936 — val_precision_w: 0.820749 — val_recall_w: 0.819948
55129/55129 [==============================] - 7s 118us/step - loss: 0.4865 - acc: 0.8365 - val_loss: 0.5466 - val_acc: 0.8199
Epoch 22/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.4820 - acc: 0.8377— val_f1: 0.729099 — val_precision: 0.795526 — val_recall: 0.689722
— val_f1_w: 0.820917 — val_precision_w: 0.825831 — val_recall_w: 0.825171
55129/55129 [==============================] - 6s 117us/step - loss: 0.4821 - acc: 0.8377 - val_loss: 0.5439 - val_acc: 0.8252
Epoch 23/30
54960/55129 [============================>.] - ETA: 0s - loss: 0.4769 - acc: 0.8401— val_f1: 0.728253 — val_precision: 0.777373 — val_recall: 0.697222
— val_f1_w: 0.816949 — val_precision_w: 0.818159 — val_recall_w: 0.820764
55129/55129 [==============================] - 7s 120us/step - loss: 0.4770 - acc: 0.8399 - val_loss: 0.5483 - val_acc: 0.8208
Epoch 24/30
54880/55129 [============================>.] - ETA: 0s - loss: 0.4726 - acc: 0.8416— val_f1: 0.737025 — val_precision: 0.785367 — val_recall: 0.705383
— val_f1_w: 0.822261 — val_precision_w: 0.824854 — val_recall_w: 0.825661
55129/55129 [==============================] - 7s 122us/step - loss: 0.4725 - acc: 0.8416 - val_loss: 0.5429 - val_acc: 0.8257
Epoch 25/30
54736/55129 [============================>.] - ETA: 0s - loss: 0.4682 - acc: 0.8432— val_f1: 0.729383 — val_precision: 0.774703 — val_recall: 0.698946
— val_f1_w: 0.819586 — val_precision_w: 0.823633 — val_recall_w: 0.821580
55129/55129 [==============================] - 6s 117us/step - loss: 0.4684 - acc: 0.8430 - val_loss: 0.5506 - val_acc: 0.8216
Epoch 26/30
54928/55129 [============================>.] - ETA: 0s - loss: 0.4644 - acc: 0.8430— val_f1: 0.719513 — val_precision: 0.784795 — val_recall: 0.681772
— val_f1_w: 0.816008 — val_precision_w: 0.819116 — val_recall_w: 0.820601
55129/55129 [==============================] - 6s 117us/step - loss: 0.4645 - acc: 0.8429 - val_loss: 0.5515 - val_acc: 0.8206
Epoch 27/30
54864/55129 [============================>.] - ETA: 0s - loss: 0.4604 - acc: 0.8444— val_f1: 0.739462 — val_precision: 0.782457 — val_recall: 0.710235
— val_f1_w: 0.823328 — val_precision_w: 0.825483 — val_recall_w: 0.826641
55129/55129 [==============================] - 7s 119us/step - loss: 0.4607 - acc: 0.8443 - val_loss: 0.5390 - val_acc: 0.8266
Epoch 28/30
54912/55129 [============================>.] - ETA: 0s - loss: 0.4567 - acc: 0.8456— val_f1: 0.726324 — val_precision: 0.789678 — val_recall: 0.689405
— val_f1_w: 0.821804 — val_precision_w: 0.826340 — val_recall_w: 0.825171
55129/55129 [==============================] - 6s 116us/step - loss: 0.4569 - acc: 0.8455 - val_loss: 0.5420 - val_acc: 0.8252
Epoch 29/30
54752/55129 [============================>.] - ETA: 0s - loss: 0.4533 - acc: 0.8479— val_f1: 0.735079 — val_precision: 0.787528 — val_recall: 0.700854
— val_f1_w: 0.824626 — val_precision_w: 0.828260 — val_recall_w: 0.828273
55129/55129 [==============================] - 6s 115us/step - loss: 0.4529 - acc: 0.8479 - val_loss: 0.5414 - val_acc: 0.8283
```

* 모델 construction과 (최고 performance를 보이는 지점까지의) training-evaluation 입니다. 매 checkpoint에서 모델들이 tutorial이라는 폴더에 저장되어야 하니, 미리 만들어 두어야겠죠 ㅎㅎ TF-IDF의 결과들이 그렇게 만족스럽지는 못했다는 걸 생각하면, 괄목할 만한 성장입니다. accuracy도 올랐고, F1의 평균값 (val_f1)도 상당한 수준으로 상승했네요. one-hot vector을 만들 때 그랬던 것처럼 그냥 구성요소들을 더했을 뿐인데 ...?

* 겨우 100차원인 벡터들을 더해서 뭘 표현할 수 있을까? 싶은 분들도 분명 계실 겁니다. 하지만, 두 가지를 상기할 필요가 있습니다. (1) word vector들은 one-hot vector들처럼 equivalent하지 않고, 특정 기준에 의해 training되었다 - 즉 그 자체로 어떤 의미를 지니고 있다. (2) 벡터들의 합으로 얻는 벡터 역시 100dim 공간에 표현될 수 있으며, 100dim은 그 방향만 해도 2^100 개 이상을 나타낼 수 있을 정도로 꽤나 많은 것을 표현할 수 있다!

---

This kind of sentence encoding gives us quite a rich representation of the sentences in the sense that the 100-dimensional vector itself yields a variety of values. This might be advantageous for tasks such as sentiment analysis, in which the inference largely relies on some polarity items or emotion words. However, we should notice that a simple summation does not say anything about the distributional or sequential information the sentence possesses; for instance, “You haven’t done it at all” and “Haven’t you done it at all” share the same word composition but their intention clearly differs. The same kind of problem is more critical in Korean, due to the language being scrambling.

* 이런 방식의 sentence vector 만들기는 sentiment classification 같은 task에서는 꽤나 좋은 성능을 냅니다. sentiment는 주로 단어 내의 어떤 polarity 및 subjectivity가 있는 단어에 의해 형성될 가능성이 높은데, 더하는 것만으로도 어떤 word가 있다는 것을 classifier가 알게 하기엔 충분하기 때문이죠. 하지만, summation의 단점은 분포나 순서를 고려할 수 없다는 겁니다. 예컨대, “You haven’t done it at all”과 “Haven’t you done it at all”은 그 단어 구성은 같아도 (capital 여부는 무시합시다) 전달하는 의미는 전혀 다르죠. word vector의 summation이 아니라 concatenation으로 한다면 이런 일을 예방할 수 있겠습니다만, 뭔가 임시방편적인 처방이고 결국 다시 저차원 임베딩이 아니게 되어버리겠죠. 이런 문제를 어떻게 하면 해결할 수 있을까요?

## 6. CNN-based sentence classification

[**Convolutional neural network**](https://ieeexplore.ieee.org/abstract/document/726791) (CNN), originally developed by LeCun, is a widely used neural network system which was primarily suggested for image processing (handwriting recognition). It roughly resembles the perception process of a visual system, summarizing a given image into an abstract values via repititive application of convolution and pooling. 

<p align="center">
    <image src="https://github.com/warnikchow/dlk2nlp/blob/master/image/alexnet2.png" width="700"><br/>
    (image from https://jeremykarnowski.wordpress.com/2015/07/15/alexnet-visualization/)

* AI를 공부하시는 분이라면 convolutional neural network, 즉 CNN을 한번쯤 들어보셨을 겁니다. 초기에 르쿤 등등에 의해 연구되고, 6-7년 전쯤부터 폭발적으로 성장하여 지금은 이미지 관련 태스크라면 베이스라인 혹은 그 베리에이션으로 인용된다는 것두요. 그러한 이력 덕분인지, NLP에 CNN을 사용한다고 하면 의아해하는 경우가 있더군요. 저도 사실 익숙해져서 그렇지, 다시 첨부터 써보라고 하면 이게 무슨 소리야 싶을지도 모르겠네요ㅎㅎ

* 제가 이미지에 CNN을 사용해본 적은 거의 없지만, 쉽게 말해 raw data를 부분부분 보고 그 정보를 추상화하는 과정을 여러번 거친다, 라고 생각하고 있습니다. 아주 러프하게요. 그것이 이미지에 처음 적용된 것이죠. 하지만 사실 정보의 추상화란 이미지에만 적용될 이유는 없습니다. 그래서 저는 cnn을 distributional information의 summarizer로 표현합니다. 어디에 무엇이 있는지 아주 간단하게 요약해 주는.

---

However, since the very classic breakthrough of [Kim 2014](https://arxiv.org/abs/1408.5882), CNN has been widely used in the text processing, understanding the word vector sequence as a single channel image. Unlike the previous approach where all the information of the words in the sentence are aggregated into a single vector, the featurization for CNN has its limitation in the volume. Thus, here we restrict the maximum length of the morpheme sequence to 30, with zero-padding for the short utterances. Taking into account the head-finality of Korean, we decided to place the word vectors on the right side of the matrix. That is, for the long utterances, only the last 30 morphemes are utilized.

```python
def featurize_cnn(corpus,wdim,maxlen):
    conv_total = np.zeros((len(corpus),maxlen,wdim,1))
    for i in range(len(corpus)):
        if i%1000 ==0:
            print(i)
        s = corpus[i]
        for j in range(len(s)):
            if s[-j-1] in model_ft and j < maxlen:
                conv_total[i][-j-1,:,0] = model_ft[s[-j-1]]
    return conv_total
    
fci_conv = featurize_cnn(fci_sp_token_train+fci_sp_token_test,100,30)
```

<p align="center">
    <image src="https://github.com/warnikchow/dlk2nlp/blob/master/image/ykim14.png" width="700"><br/>
    (image from Kim 2014)

* 이것이 문장 분류에 어떻게 사용되느냐? 가장 먼저 거치는 과정은 쉽게 말해 문장을 그림처럼 바꾸는 겁니다. 즉, 단일 채널 matrix를 만드는 거죠 (그림은 보통 rgb의 3 channel). 우린 sentence matrix란 걸 논한 적 없으니 word vector들로 어떻게 해 봐야 될 텐데, word vector나 TF-IDF를 가지고는 듬성듬성하게 nonzero가 박혀 있는 것들밖에 만들지 못할 테죠. 애초에 값에 대한 위치 bias가 없는 녀석들이니 순서(order)적인 것 외에 아무 정보도 CNN에 주지는 못할 겁니다.

* 이 때 다시 등장하는 것이 앞서 언급한 word2vec입니다. 문장을 수치화해 넣을 수 있는 일종의 고정된 사이즈의 도화지가 있다고 생각해 봅시다. 예컨대 100 x 30정도의? 거기에 100-dim word vector 30개를 padding해 넣는 겁니다. 물론 문장 길이가 30이 되지 않을 수도 있지요. 그러면 빈 부분은 0으로 채웁니다. 진짜 없으니까요. 문장이 더 길다면? 자릅니다. 물론 이 부분은 '문장 최대길이'를 조사해서 적절히 설정하면 될 일입니다 (물론 이렇게 하지 않고 모두 보존하는 방법도 있겠습니다만, 일단 여기선 다루지 않겠습니다). 이 과정에서 한국어의 head-finality를 고려하여 문장은 오른쪽에 치우치게 배열하도록 결정하였습니다. 한국어는 역시 끝까지 들어봐야 하니까, 자르더라도 끝은 남겨야죠!

---

There are so many types of convolutional networks out there (LeNet, AlexNet, VGG, YOLO ...), however such deep and wide networks might not be required for the sentence classification. The model architecture used for the implementation is quite simple; two convolutional layers with the window width 3 and a max pooling layer between with the size (2,1). For the first conv-layer, the window size is (3,100) and for the second (3,1), since the information was abstracted and max-pooled to make up a single vector.

```python
def validate_cnn(result,y,filters,hidden_dim,cw,filename):
    model = Sequential()
    model.add(layers.Conv2D(filters,(3,len(result[0][0])),activation= 'relu',input_shape = (len(result[0]),len(result[0][0]),1)))
    model.add(layers.MaxPooling2D((2,1)))
    model.add(layers.Conv2D(filters,(3,1),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(hidden_dim,activation='relu'))
    model.add(layers.Dense(int(max(y)+1), activation='softmax'))
    model.summary()
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    model.fit(result,y,validation_split=0.1,epochs=30,batch_size=16,callbacks=callbacks_list,class_weight=cw)

validate_cnn(fci_conv,fci_label,32,128,class_weights_fci,'model/tutorial/conv')
```

* 일단 image를 cnn에 적용하는 과정을 패러미터화하면, 채널, 필터, 컨벌루션레이어, 윈도우, 풀링 정도로 요약할 수 있습니다. 채널은 앞서 말했듯 rgb 같이 몇 개의 요소로 나타내냐이며 필터는 얼마나 병렬로 처리할거냐, 컨벌루션레이어 수는 추상화 과정을 몇번 거칠거냐, 윈도우는 어떤 식으로 각 컨벌루션 레이어를 훑을거냐, 풀링은 컨벌루션 레이어를 훑은 값들에서 중요한 요소들을 어떻게 취사선택할거냐? 이정도로 나타낼 수 있겠네요.

* 이미지의 cnn은 그래서 일반적으로 3채널, 많은 필터 (>64?), 다층 컨벌루션 레이어, 상하좌우로 stride되는 3 by 3 혹은 5 by 5 window 등으로 요약될 수 있습니다. 물론 alexnet, vgg, yolo 등 다양한 아키텍쳐들이 있고, 모두 특색이 있겠지만, 기본적으론 저렇습니다. 하지만 word vector sequence에서 상하좌우로 움직이는 window가 어떤 의미가 있을까요? 우리는 100dim의 벡터 각 엔트리에 어떤 성질의 성분들이 자리잡고 있는지 알지 못하며, 굳이 그런 성질을 지정해줄 필요도 느끼지 못하였습니다. 이미지는 두차원 모두가 semantic을 포함하지만, sentence에서 semantic이 의미가 있는 방향은 word vector가 pad되는 방향이니까요.

* 그래서 저는 sentence의 cnn에선 3 by 100 혹은 5 by 100 window를 사용합니다. 결론적으론 2D convolution이 1D처럼 돼버리긴 합니다만, 역설적으로 문장을 그림이 아니라 문장처럼 볼 수 있는 방법이 되는 것 같아요. Word2vec의 결과물로 나온 그 단어의 vector의 특색을 결정짓는 entry를 가장 왜곡 없이 전달해줄 수 있는 방법이라고 저는 보고 있습니다. 그 이후의 max-pooling과 추가적인 convolution 아키텍쳐는 개인의 선택에 달렸지만요. Boolean distributional semantics처럼 word vector의 각 entry가 어떤 의미를 갖는다면 모르겠지만, 그렇지 않다면 문장은 문장처럼 읽는 것이 cnn에 있어서도 효과적이지 않나? 라는 것이 저의 생각입니다. (물론 반례 및 피드백은 언제나 환영입니다!)

---

The result is encouraging! Although there was a little improvement in F1 score, we had about 20% RRER for the accuracy (regarding the model with the best performance). The CNN-based featurization and classification of the sentence shows quite satisfactory result with very fast training. However, the architecture does not seem to still convey the correlation between the non-consecutive components. The recurrent neural network covers such characteristics, with a sequence-based approach.

```properties
# CONSOLE RESULT
>>> validate_cnn(fci_conv,fci_label,32,128,class_weights_fci,'model/tutorial/conv')
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 1, 32)         9632      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 1, 32)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 1, 32)         3104      
_________________________________________________________________
flatten_1 (Flatten)          (None, 384)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 128)               49280     
_________________________________________________________________
dense_6 (Dense)              (None, 7)                 903       
=================================================================
Total params: 62,919
Trainable params: 62,919
Non-trainable params: 0
_________________________________________________________________
Train on 55129 samples, validate on 6126 samples
Epoch 1/30
54992/55129 [============================>.] - ETA: 0s - loss: 0.5817 - acc: 0.8026— val_f1: 0.715119 — val_precision: 0.766993 — val_recall: 0.687496
— val_f1_w: 0.836404 — val_precision_w: 0.840650 — val_recall_w: 0.842148
55129/55129 [==============================] - 21s 375us/step - loss: 0.5816 - acc: 0.8027 - val_loss: 0.4907 - val_acc: 0.8421
Epoch 2/30
54992/55129 [============================>.] - ETA: 0s - loss: 0.4391 - acc: 0.8505— val_f1: 0.748518 — val_precision: 0.785899 — val_recall: 0.724397
— val_f1_w: 0.850521 — val_precision_w: 0.852202 — val_recall_w: 0.853901
55129/55129 [==============================] - 21s 376us/step - loss: 0.4389 - acc: 0.8506 - val_loss: 0.4531 - val_acc: 0.8539
Epoch 3/30
54992/55129 [============================>.] - ETA: 0s - loss: 0.3934 - acc: 0.8667— val_f1: 0.755061 — val_precision: 0.791092 — val_recall: 0.731512
— val_f1_w: 0.855977 — val_precision_w: 0.857298 — val_recall_w: 0.859288
55129/55129 [==============================] - 20s 371us/step - loss: 0.3936 - acc: 0.8667 - val_loss: 0.4413 - val_acc: 0.8593
Epoch 4/30
54992/55129 [============================>.] - ETA: 0s - loss: 0.3640 - acc: 0.8753— val_f1: 0.762634 — val_precision: 0.783417 — val_recall: 0.755429
— val_f1_w: 0.853230 — val_precision_w: 0.858127 — val_recall_w: 0.856024
55129/55129 [==============================] - 21s 374us/step - loss: 0.3640 - acc: 0.8753 - val_loss: 0.4558 - val_acc: 0.8560
Epoch 5/30
54992/55129 [============================>.] - ETA: 0s - loss: 0.3402 - acc: 0.8824— val_f1: 0.756524 — val_precision: 0.798401 — val_recall: 0.728596
— val_f1_w: 0.857324 — val_precision_w: 0.857582 — val_recall_w: 0.860431
55129/55129 [==============================] - 20s 370us/step - loss: 0.3403 - acc: 0.8824 - val_loss: 0.4348 - val_acc: 0.8604
Epoch 6/30
55024/55129 [============================>.] - ETA: 0s - loss: 0.3210 - acc: 0.8910— val_f1: 0.761926 — val_precision: 0.782188 — val_recall: 0.748026
— val_f1_w: 0.856406 — val_precision_w: 0.857747 — val_recall_w: 0.858146
55129/55129 [==============================] - 21s 374us/step - loss: 0.3209 - acc: 0.8910 - val_loss: 0.4406 - val_acc: 0.8581
Epoch 7/30
55040/55129 [============================>.] - ETA: 0s - loss: 0.3051 - acc: 0.8947— val_f1: 0.758700 — val_precision: 0.784020 — val_recall: 0.740566
— val_f1_w: 0.850813 — val_precision_w: 0.850469 — val_recall_w: 0.853575
55129/55129 [==============================] - 21s 373us/step - loss: 0.3051 - acc: 0.8947 - val_loss: 0.4540 - val_acc: 0.8536
Epoch 8/30
55088/55129 [============================>.] - ETA: 0s - loss: 0.2900 - acc: 0.9003— val_f1: 0.764175 — val_precision: 0.814364 — val_recall: 0.732869
— val_f1_w: 0.861861 — val_precision_w: 0.865253 — val_recall_w: 0.865491
55129/55129 [==============================] - 21s 373us/step - loss: 0.2899 - acc: 0.9004 - val_loss: 0.4441 - val_acc: 0.8655
```

* 결과입니다. F1 score에서는 그렇게 큰 향상을 얻지는 못했으나, accuracy는 0.8283에서 0.8655로, error rate가 약 17%에서 14%로 감소하며 약 20%의 에러율 감소 (RRER)을 보였습니다. 아무래도 distributional한 information을 반영하는 것이, 단순히 existence를 체크하는 것보다는 더 정확하겠죠.

* 이상으로 distributional information의 가장 효과적인 summarizer 중 하나에 대하여 살펴봤습니다. 하지만 그 단점은, non-consecutive한 성분들 간의 상관관계를 설명하기 쉽지 않다는 점이죠. 다음 편부터 설명되는 rnn은 그러한 점들을 보완합니다.

## 7. RNN (BiLSTM)-based sentence classification

**Recurrent neural network (RNN)**, which was suggested originally in the late 20th century, is a representative network that reflects the sequential information in the numerical summarization. Due to high-computation issue, its materialization has recently been possible with the help of modern computing systems (e.g. GPU boost-up). The problem of vanishing gradient has been partially solved by [**long short-term memory (LSTM)**](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735), whose direction-bias was improved along with the bidirectional sequencing (BiLSTM).

<p align="center">
    <image src="https://github.com/warnikchow/dlk2nlp/blob/master/image/rnn.jpg" width="700"><br/>
    (image from https://aikorea.org/blog/rnn-tutorial-1/)

* RNN, 즉 recurrent neural network란 time-series의 input을 받아, 그에 대한 latent information을 담고 있는 hidden layer sequence를 생성하되, 특정 시점에서의 hidden layer가 그 시점에서의 input data와 이전 시점의 hidden layer로부터 연산되는 알고리듬입니다. hidden Markov model (HMM)과 그 컨셉은 유사하지만 바로 이전 시점에만 영향받지 않고, 앞서 있는 데이터 모두의 정보를 뒷부분의 hidden layer에 반영한다는 장점이 있죠. 쉽게 말해 sequential data의 summarization이라고 보면 될 것 같네요.

* rnn을 트레이닝하는 과정 역시 일반적인 mlp나 cnn에서와 마찬가지로 back-propagation을 이용하게 되는데요, 이 과정에서 vanishing gradient의 문제가 발생하게 됩니다. 너무 많은 정보들이 encoding되다 보니 패러미터가 폭발해 버리는 겁니다. 사람은 이와 다르게, 문장이나 글이 길어지게 되면 너무 멀리 떨어져 있는 정보는 잊어버리죠 :D 뭐 그게 꼭 좋은 건 아니겠지만서두, 정보 과잉을 방지해주거나 뭐 그렇지 않겠습니까? 그런 컨셉으로 나온 것이 적당히 forget gate를 추가한, 1997년의 long short-term memory (LSTM) 입니다. lstm이 앞부분의 정보를 반영하지 못한다는 단점을 보완하기 위해 제시된 것이, lstm을 양방향으로 (처음에서 시작해서 끝으로, 끝에서 시작해서 처음으로) 하여 얻은 hidden layer sequences를 augment한 Bidirectional lstm도 같은 해에 제시되었구요. 생각보다 옛날인데 왜 이제 와서 유행하게 됐냐구요? 계산량이 엄청나기 때문이죠... 갓비디아 짱짱컴퍼니

---

In this tutorial, we utilize the BiLSTM structure. The featurization is almost identical to the case of CNN, except that the channel information is omitted. To say short, for the same data (ignoring the channel number), CNN extracts locally notable features and RNN extracts the relations reflected in the sequential arrangement.

```python
def featurize_rnn(corpus,wdim,maxlen):
    rnn_total = np.zeros((len(corpus),maxlen,wdim))
    for i in range(len(corpus)):
        if i%1000 ==0:
            print(i)
        s = corpus[i]
        for j in range(len(s)):
            if s[-j-1] in model_ft and j < maxlen:
                rnn_total[i][-j-1,:] = model_ft[s[-j-1]]
    return rnn_total

fci_rec = featurize_rnn(fci_sp_token_train+fci_sp_token_test,100,30)

from keras.layers import LSTM
from keras.layers import Bidirectional

def validate_bilstm(result,y,hidden_lstm,hidden_dim,cw,filename):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_lstm), input_shape=(len(result[0]), len(result[0][0]))))
    model.add(layers.Dense(hidden_dim, activation='relu'))
    model.add(layers.Dense(int(max(y)+1), activation='softmax'))
    model.summary()
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    model.fit(result,y,validation_split=0.1,epochs=30,batch_size=16,callbacks=callbacks_list,class_weight=cw)

validate_bilstm(fci_rec,fci_label,32,128,class_weights_fci,'model/tutorial/rec')
```

* 본 튜토리얼에서는 가장 무난히 적용할 수 있는 BiLSTM을 사용합니다. LSTM의 문제는 문장이 길어질수록 앞부분의 내용을 반영하지 못할 수 있다는 것인데, BiLSTM은 LSTM을 양방향으로 돌린 두 sequence를 hidden layer sequence 하나로 concatenate하면서 어느 정도 그런 점을 보완할 수 있게 합니다. CNN과 간단히 차이를 말해 보자면, CNN은 global하게 본 후 local하게 중요한 피쳐들을 뽑아내어 추상화시키는 것이고, RNN (BiLSTM)은 sequential한 배열에 내재한 관계를 추상화한다고 보면 될 것 같습니다. 

---

We obtained another boost in performance, especially very high in F1 score. This implies that our task which deals with the intention of Korean sentences is largely influenced by the word order. This may have been an important factor in identifying the rhetoricalness, for instance, '뭐 해 지금 (what / do / now, *what (on earth) are you doing now*)' sounds much more rhetorical than '지금 뭐 해 (now / what / do, *what are you doing now*)', in view of native. However, so far we've conducted the experiments given the morpheme-level decomposition. Will the real semantics NOT be embedded in the characters? Well, before that, how should **character** be defined in Korean, which has distinguished morpho-syllabic blocks (which are different from alphabets) as a unit of character? 

```properties
# CONSOLE RESULT
>>> validate_bilstm(fci_rec,fci_label,32,128,class_weights_fci,0.1,16,'model/tutorial/rec')
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_1 (Bidirection (None, 64)                34048     
_________________________________________________________________
dense_7 (Dense)              (None, 128)               8320      
_________________________________________________________________
dense_8 (Dense)              (None, 7)                 903       
=================================================================
Total params: 43,271
Trainable params: 43,271
Non-trainable params: 0
_________________________________________________________________
Train on 55129 samples, validate on 6126 samples
Epoch 1/50
55088/55129 [============================>.] - ETA: 0s - loss: 0.5545 - acc: 0.8145— val_f1: 0.732477 — val_precision: 0.764964 — val_recall: 0.716559
— val_f1_w: 0.838581 — val_precision_w: 0.844676 — val_recall_w: 0.841169
55129/55129 [==============================] - 86s 2ms/step - loss: 0.5544 - acc: 0.8145 - val_loss: 0.4835 - val_acc: 0.8412
Epoch 2/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.4302 - acc: 0.8532— val_f1: 0.755420 — val_precision: 0.818620 — val_recall: 0.719585
— val_f1_w: 0.856714 — val_precision_w: 0.860478 — val_recall_w: 0.860757
55129/55129 [==============================] - 83s 2ms/step - loss: 0.4302 - acc: 0.8531 - val_loss: 0.4248 - val_acc: 0.8608
Epoch 3/50
55088/55129 [============================>.] - ETA: 0s - loss: 0.3909 - acc: 0.8658— val_f1: 0.769535 — val_precision: 0.820192 — val_recall: 0.738047
— val_f1_w: 0.860804 — val_precision_w: 0.862253 — val_recall_w: 0.864675
55129/55129 [==============================] - 82s 1ms/step - loss: 0.3908 - acc: 0.8658 - val_loss: 0.4071 - val_acc: 0.8647
Epoch 4/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.3626 - acc: 0.8757— val_f1: 0.780424 — val_precision: 0.814412 — val_recall: 0.758550
— val_f1_w: 0.868592 — val_precision_w: 0.869740 — val_recall_w: 0.870389
55129/55129 [==============================] - 82s 1ms/step - loss: 0.3626 - acc: 0.8758 - val_loss: 0.3861 - val_acc: 0.8704
Epoch 5/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3419 - acc: 0.8823— val_f1: 0.783106 — val_precision: 0.806826 — val_recall: 0.767035
— val_f1_w: 0.871497 — val_precision_w: 0.871639 — val_recall_w: 0.872837
55129/55129 [==============================] - 83s 2ms/step - loss: 0.3418 - acc: 0.8823 - val_loss: 0.3835 - val_acc: 0.8728
Epoch 6/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.3238 - acc: 0.8889— val_f1: 0.787453 — val_precision: 0.815045 — val_recall: 0.766895
— val_f1_w: 0.872195 — val_precision_w: 0.874534 — val_recall_w: 0.874633
55129/55129 [==============================] - 82s 1ms/step - loss: 0.3239 - acc: 0.8889 - val_loss: 0.3798 - val_acc: 0.8746
Epoch 7/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3093 - acc: 0.8935— val_f1: 0.782065 — val_precision: 0.797697 — val_recall: 0.772691
— val_f1_w: 0.873711 — val_precision_w: 0.875246 — val_recall_w: 0.873327
55129/55129 [==============================] - 82s 1ms/step - loss: 0.3093 - acc: 0.8935 - val_loss: 0.3732 - val_acc: 0.8733
Epoch 8/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2953 - acc: 0.8978— val_f1: 0.777524 — val_precision: 0.800048 — val_recall: 0.762805
— val_f1_w: 0.867708 — val_precision_w: 0.868467 — val_recall_w: 0.869572
55129/55129 [==============================] - 83s 1ms/step - loss: 0.2952 - acc: 0.8978 - val_loss: 0.3942 - val_acc: 0.8696
Epoch 9/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2820 - acc: 0.9028— val_f1: 0.783454 — val_precision: 0.822934 — val_recall: 0.758621
— val_f1_w: 0.875902 — val_precision_w: 0.877293 — val_recall_w: 0.879367
55129/55129 [==============================] - 82s 1ms/step - loss: 0.2820 - acc: 0.9028 - val_loss: 0.3732 - val_acc: 0.8794
Epoch 10/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2708 - acc: 0.9065— val_f1: 0.790558 — val_precision: 0.811155 — val_recall: 0.773668
— val_f1_w: 0.877717 — val_precision_w: 0.876924 — val_recall_w: 0.879530
55129/55129 [==============================] - 82s 1ms/step - loss: 0.2708 - acc: 0.9065 - val_loss: 0.3804 - val_acc: 0.8795
Epoch 11/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2572 - acc: 0.9108— val_f1: 0.778422 — val_precision: 0.789590 — val_recall: 0.770837
— val_f1_w: 0.870727 — val_precision_w: 0.872009 — val_recall_w: 0.872674
55129/55129 [==============================] - 82s 1ms/step - loss: 0.2571 - acc: 0.9108 - val_loss: 0.3960 - val_acc: 0.8727
Epoch 12/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2460 - acc: 0.9154— val_f1: 0.788357 — val_precision: 0.805196 — val_recall: 0.775248
— val_f1_w: 0.876743 — val_precision_w: 0.876289 — val_recall_w: 0.878387
55129/55129 [==============================] - 82s 1ms/step - loss: 0.2460 - acc: 0.9154 - val_loss: 0.3865 - val_acc: 0.8784
Epoch 13/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2356 - acc: 0.9185— val_f1: 0.768287 — val_precision: 0.766577 — val_recall: 0.772194
— val_f1_w: 0.864052 — val_precision_w: 0.865701 — val_recall_w: 0.864022
55129/55129 [==============================] - 82s 1ms/step - loss: 0.2355 - acc: 0.9186 - val_loss: 0.4198 - val_acc: 0.8640
Epoch 14/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2253 - acc: 0.9225— val_f1: 0.768811 — val_precision: 0.775967 — val_recall: 0.768394
— val_f1_w: 0.864381 — val_precision_w: 0.867127 — val_recall_w: 0.864185
55129/55129 [==============================] - 82s 1ms/step - loss: 0.2254 - acc: 0.9225 - val_loss: 0.4194 - val_acc: 0.8642
Epoch 15/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2147 - acc: 0.9258— val_f1: 0.782913 — val_precision: 0.790473 — val_recall: 0.778720
— val_f1_w: 0.875200 — val_precision_w: 0.876058 — val_recall_w: 0.875775
55129/55129 [==============================] - 82s 1ms/step - loss: 0.2147 - acc: 0.9258 - val_loss: 0.4084 - val_acc: 0.8758
Epoch 16/50
55088/55129 [============================>.] - ETA: 0s - loss: 0.2045 - acc: 0.9285— val_f1: 0.790696 — val_precision: 0.819661 — val_recall: 0.768516
— val_f1_w: 0.878328 — val_precision_w: 0.878134 — val_recall_w: 0.880673
55129/55129 [==============================] - 82s 1ms/step - loss: 0.2044 - acc: 0.9285 - val_loss: 0.4099 - val_acc: 0.8807
```

* 놀랍게도 정확도가 다시 한번 올랐고, F1 score가 상당한 수준으로 올랐습니다. 평균치가 저 정도 올랐다는 것은, 이제 아무리 F1 score가 안 좋아도 0.5 정도는 된다는 걸 의미한다고 봐도 무방할 것 같습니다. 처음에 TF-IDF로 코드를 돌릴 때 intonation-dependent utterances (마지막 case) 에 대해 상당히 낮은 F1 score가 나왔던 것 같은데, 해당 태스크에서 상대적으로 Recall (재현률)을 높게 하여 false alarm을 울리는 것이 false positive보다는 낫다는 점을 고려하면 나쁘지 않은 결과입니다. 그런데 한 가지 간과한 것이 있습니다. 지금까지 우리는 열심히 morpheme들을 가지고 놀았는데, 과연 이것을 character-level로 끊어 보면 어떻게 될까요? 아니, 그 전에 우선, 한국어에서 character을 어떻게 정의하는 것이 바람직할까요?

## 8. Character embedding

Due to the distinguished writing style, the embedding of Korean letters *Hangul* is difficult even from its definition. For many Romanian or German langauges, the letters of latin alphabet are utilized as character and thereby character-level embeddding has been widely used in the region of text analysis since [Zhang, 2015](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classifica). However, for Korean, the morpho-syllabic blocks that represent the syllables, can be decomposed into *jamo*, the alphabet of the Korean writing system.

To be specific, The letters of alphabet make up the morpho-syllabic blocks (characters) that are equal to the phonetic unit of the syllable, in the conjunct form of Syllable: CV(C). This notation implies that there should be at least one consonant (namely *cho-seng*, the first sound) and one vowel (namely *cwung-seng*, the second sound). An additional consonant (namely *cong-seng*, the third sound) is auxiliary. However, usually in the character decomposition, three slots are fully held for each component; an empty cell comes for the third entry if there is no auxiliary consonant. The number of the possible letters, or (composite) consonants/vowels, that can come for each slot is 19, 21, and 27. For instance, in a character '각 (*kak*)', three clock-wisely arranged alphabets ㄱ, ㅏ, and ㄱ, each sounds *k*, *a*, and *k* respectively.

* 앞서 말했듯, 한국어의 writing system은 자연발생된 것이 아닌 단체 (거진 개인)에 의해 창조되었다는 점에서 매우 특별하며, 어떤 음절에 해당하는 morpho-syllabic block을 decompose하여 그 음절을 이루는 소리 성분들에 해당하는 sub-character, 즉 자모를 알아낼 수 있다는 점에서도 독특합니다.

* 좀 더 자세히 얘기하자면 한글은 CV(C)의 conjunct form으로 되어 있으며, 이는 한 음절을 구성하기 위해 최소한 자모 한 개씩은 있어야 함을 의미합니다. 이는 초성과 중성이라고 불리며, CV를 지칭하죠. 종성, 즉 third sound (C)의 경우는, 있어도 되고 없어도 상관없습니다. 초성으로 가능한 자음은 19개, 중성으로 가능한 모음은 21개이며, 종성으로 가능한 자음은 composite consonants를 포함하여 27개입니다 (empty일 경우 포함하면 28개).

---

For a clearness, let's denote the morpho-syllabic blocks as *characters* and the consonants/vowels as *Jamo*. The description above yields total 11,172 possible characters. However, one-hot encoding those combinations into 11,172-dim vector seems very redundant at a glance. Therefore, there have been various character encoding schemes suggested. The schemes include two approaches; (1) decomposing the blocks into sub-characters and (2) preserving them.

* 보다 논의를 명확히 하기 위해, 음절들을 character라고 하고 자모를 Jamo라고 둡시다. 위의 설명에 따르면 우리는 총 11,172개의 가능한 자모 조합을 얻는데, 이를 one-hot encoding하는 것은 매우 costly해 보이기도 하거니와 그에 따른 merit을 딱히 찾기도 어렵습니다. 따라서 이 문제에 대하여 한국어 character을 encoding하는 다양한 방법들이 소개되었는데, 이 방법들은 크게 1) 자소 분리 2) 음절 유지의 두 가지로 나뉠 수 있습니다.

---

For the first approach, the most simple way is to spread a character into three alphabet sequence. There are some toolkits (e.g., [hgtk](https://github.com/bluedisk/hangul-toolkit)) that decompose Hangul characters into alphabets; by utilizing them, we can obtain the tuple ('ㄱ', 'ㅏ', 'ㄱ') from a single character '각'. This blurs the property of full characters but allows a sparse representation with low dimensional vectors (of size 67). However note that the total length of numericalization increases since the sole alphabets ('ㄱ' and 'ㅏ') and the characters ('각') are assigned equally two bytes. Due to this, degrade in computation efficiency is inavoidable. To deal with this, [romanization has been suggested and showed a good performance](https://arxiv.org/abs/1708.02657), but here we don't utilize it since the processing may induce an ambiguity. 

In this tutorial, we present two conventional methodologies in one-hot encoding, namely [Shin](https://www.dbpia.co.kr/Journal/ArticleDetail/NODE07207314#) and [Cho](http://www.dbpia.co.kr/Journal/ArticleDetail/NODE07503227). The former is a simple 67-dim version, and the latter considers the cases where the alphabets are used solely (e.g., 'ㅠㅠ', 'ㅋㅋ'), not in a form of block. Each corresponds with *shin_onehot* and *cho_onehot* in the code below. A description on *char2onehot* will be provided afterwards.

```python
import hgtk
from hgtk.letter import decompose as decom

choseng = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㄲ','ㄸ','ㅃ','ㅆ','ㅉ']
cwungseng = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅘ','ㅙ','ㅚ','ㅝ','ㅞ','ㅟ','ㅢ']
congseng = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㄲ','ㅆ','ㄳ','ㄵ','ㄶ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅄ
']
alp = choseng+cwungseng+congseng
uniquealp = list(set(choseng+cwungseng+congseng))

def cho2onehot(s):
    res = np.zeros(len(choseng))
    if s in choseng:
        res[choseng.index(s)]=1
    return res

def cwu2onehot(s):
    res = np.zeros(len(cwungseng))
    if s in cwungseng:
        res[cwungseng.index(s)]=1
    return res

def con2onehot(s):
    res = np.zeros(len(congseng))
    if s in congseng:
        res[congseng.index(s)]=1
    return res

def uni2onehot(s):
    res = np.zeros(len(uniquealp))
    if s in uniquealp:
        res[uniquealp.index(s)]=1
    return res
    
def shin_onehot(s):
    z = decom(s)
    res = np.zeros((len(alp),3))
    res[:len(choseng),0] = cho2onehot(z[0])
    res[len(choseng):len(choseng)+len(cwungseng),1] = cwu2onehot(z[1])
    res[len(choseng)+len(cwungseng):len(alp),2] = con2onehot(z[2])
    return res

def cho_onehot(s):
    z = decom(s)
    res = np.zeros((len(alp)+len(uniquealp),3))
    if len(z[0]+z[1]+z[2]) > 1:
        res[:len(alp),:] = shin_onehot(s)
    elif len(z[0])>0:
        res[len(alp):,0] = uni2onehot(s)
    elif len(z[1])>0:
        res[len(alp):,1] = uni2onehot(s)
    else:
        res[len(alp):,2] = uni2onehot(s)
    return res
    
def char2onehot(s):
    z = decom(s)
    res = np.concatenate([cho2onehot(z[0]),cwu2onehot(z[1]),con2onehot(z[2])])
    return res
```

* 자소 분리의 방법에 있어 가장 먼저 생각해볼 수 있는 것은 한 글자 (character)을 세 개의 자모 sequence (e.g., '각' > 'ㄱ', 'ㅏ', 'ㄱ')로 나타내는 것입니다. 이를 손쉽게 수행할 수 있는 [hgtk](https://github.com/bluedisk/hangul-toolkit)와 같은 툴킷들도 제공되고 있구요. 이러한 변환이 full character (syllabic block)의 property를 뚜렷하게 하지 못할 수 있으나, low dimension(67?)의 sparse representation을 가능하게 한다는 점에서 유용할 수 있습니다. 다만 이 과정에서 정보량이 세 배로 증가해서 computation efficiency를 떨어뜨릴 수 있다는 점은 단점이 될 수 있겠네요. 이를 해결할 수 있는 방법으로 romanized Hangul을 생각해볼 수 있겠지만, 직관적으로 어색하고 중의성을 유발할 수 있기에 여기서는 따로 다루지 않겠습니다.

* 본 튜토리얼에서 다루는 one-hot encoding방법은 [Shin](https://www.dbpia.co.kr/Journal/ArticleDetail/NODE07207314#) 과 [Cho](http://www.dbpia.co.kr/Journal/ArticleDetail/NODE07503227)의 두 가지입니다. 전자는 앞서 말한 67-dim의 단순 임베딩이며, 후자는 자모가 단독으로 사용되는 경우들을 고려하여 unique하게 사용되는 것을 지칭하는 별도의 벡터를 augment해 줍니다. 물론 해당 논문들에서는 특수 기호들에 대한 placeholder들도 고려하지만, 3i4k데이터셋에는 punctuation이 지워져 있기 때문에 각각 67/118 dim을 쓰는 것으로 충분하다고 보도록 하겠습니다. *char2onehot*에 대한 설명은 다음 단락에서 이어가도록 하겠습니다.

---

For the second type of approachs, namely block-preserving ones, a significant advantage is that we can treat the characters as subword, which can be detered in the previous approaches. Theoretically we can utilize 11,172 characters as a syllable, but in real life it is sufficient with only around 2,500 ones. This allows us to utilize the one-hot encoding of the syllables. A simple integer-indexed dictionary that contains 2,534 syllables from [100-dimension fastText vector dictionary which was trained with 2M drama scripts](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor) is uploaded here. However, another practical approach that can be adopted is using the dense character vectors from the fastText dictionary we've embedded. It compresses a sparse 2,534-dim vector into a dense 100-dim vector, possibly reflecting the distributive semantics of syllables as subwords of morphemes. 

Besides, another sparse embedding scheme that aggregates all sounds in a block into a form of multi-hot encoding, was suggested recently, and we call it [Song](https://www.researchgate.net/publication/331987503_Sequence-to-Sequence_Autoencoder_based_Korean_Text_Error_Correction_using_Syllable-level_Multi-hot_Vector_Representation/stats). The code below imports the dictionary for sparse syllable encoding, and the multi-hot encoded vectors can be obtained by *char2onehot* defined above.

```python
kor_char = np.load('kor_char.npy').item()
```

* 그 다음으로 자소 분리를 하지 않고 음절을 임베딩하는 것을 생각해볼 수 있는데요, 가장 큰 장점은 일종의 subword로써의 음절의 성질을 유지할 수 있다는 점입니다. 물론 형태소가 '-ㄴ', '-ㄹ'처럼 decompose되어 나타나는 경우도 있지만, 앞서 말했듯 우리는 non-invasive한 형태소 분석기를 쓸 거니까요. 위에 언급했듯 가능한 음절 조합은 11,172개이지만 실생활에서 사용되는 가짓수는 대략 2,500개 정도이며 이는 NN classification들에서 사용했던 fastText dictionary에 존재하는 음절의 갯수가 2,534개임을 고려해 볼 때 어느 정도 타당한 주장임을 알 수 있습니다. 그래서 가장 먼저 생각해볼 수 있는 one-hot encoding 방법은 해당 dictionary에 있는 모든 음절에 번호를 매겨 사용하는 것이죠. 물론 2,534도 작은 수는 아닙니다만, 모두 사용되는지조차 알 수 없는 11,172개의 음절을 모두 가정하는 것보단 훨씬 효율적이리라 예측할 수 있지요. 해당 음절 모음은 위의 code를 통해 import할 수 있습니다.

* 또다른 음절 임베딩 방법으로는, 언급했던 fastText dictionary에 있는 length 1의 단어들 (음절들)의 dense embedding만 활용하는 것을 생각해볼 수 있겠습니다. 이는 2,534의 크기를 100으로 줄이면서, 음절들 간의 distributional semantics도 고려해줄 수 있기에 상당한 성능 개선이 있을 것으로 예상해볼 수 있는 방법입니다. 실제로 제가 [딥러닝 기반 띄어쓰기](https://github.com/warnikchow/ttuyssubot)에서 활용하여 어느 정도의 효과를 보았지요. 

* 그리고 또 하나의 임베딩 방법이 최근 제시되었는데, 바로 하나의 character의 초/중/종성을 multi-hot vector로 만들어 주는 방법입니다. 저는 그런 생각을 한 게 제가 처음인 줄 알고 일단 특허 쓰고 논문을 작성 중이었는데, 구글 스칼라에서는 검색되지 않지만 이미 2018년 HCLT에서 [같은 방법](https://www.researchgate.net/publication/331987503_Sequence-to-Sequence_Autoencoder_based_Korean_Text_Error_Correction_using_Syllable-level_Multi-hot_Vector_Representation/stats)이 제시가 되었더라구요 ㅎㅎ역시 이 세계는 생각나는 대로 바로바로 하지 않으면 누가 하는 곳... 어쨌든 그래서 그 방식도 함께 여기서 비교해 보도록 하겠습니다. 해당  위에 있는 *char2onehot*을 통해 정의됩니다.

---

The aforementioned methodologies are implemented in the following code as a featurization for RNN. Since the number of morphemes are generally smaller than the number of 솓 characters, we fixed the max character length to 80. Consequently, the sequence length for the *Jamo*-level embeddings (Shin & Cho) reaches 240. Again, to reflect the head-finality of Korean, the Jamos/characters are padded from the sentence-final.

```python
def featurize_rnnchar(corpus,wdim,chardict,maxlen):
    rnn_shin  = np.zeros((len(corpus),maxlen*3,len(alp)))
    rnn_cho   = np.zeros((len(corpus),maxlen*3,len(alp)+len(uniquealp)))
    rnn_char  = np.zeros((len(corpus),maxlen,len(alp)))
    rnn_onehot= np.zeros((len(corpus),maxlen,len(chardict)))
    rnn_total = np.zeros((len(corpus),maxlen,wdim))
    for i in range(len(corpus)):
        if i%1000 ==0:
            print(i)
        s = corpus[i]
        for j in range(len(s)):
            if j < maxlen and hgtk.checker.is_hangul(s[-j-1])==True:
                if j>0:
                    rnn_shin[i][-3*j-3:-3*j,:] = np.transpose(shin_onehot(s[-j-1]))
                    rnn_cho[i][-3*j-3:-3*j,:] = np.transpose(cho_onehot(s[-j-1]))
                else:
                    rnn_shin[i][-3*j-3:,:] = np.transpose(shin_onehot(s[-j-1]))
                    rnn_cho[i][-3*j-3:,:] = np.transpose(cho_onehot(s[-j-1]))
                rnn_char[i][-j-1,:] = char2onehot(s[-j-1])
                if s[-j-1] in model_ft:
                    rnn_total[i][-j-1,:] = model_ft[s[-j-1]]
                if s[-j-1] in chardict:
                    rnn_onehot[i][-j-1,chardict[s[-j-1]]]=1
    return rnn_shin, rnn_cho, rnn_char, rnn_onehot, rnn_total

fci_rec_shin, fci_rec_cho, fci_rec_char, fci_rec_onehot, fci_rec = featurize_rnnchar(fci_data,100,kor_char,80)
```

* 앞서 얘기한 character embedding 방법론들이 RNN featurization의 형태로 정리된 코드입니다. 각각 Shin, Cho, Song, one-hot encoded char embedding, 그리고 dense char embedding 을 output으로 합니다. 보통 형태소의 개수보다 음절 개수가 훨씬 많기 때문에 길이는 80으로 하였고, 이 과정에서 space도 하나의 character로 카운트됩니다. 자소분리 임베딩의 경우 3배의 길이를 가지는 것으로 간주하여 size 240을 최대로 하였습니다. 그리고 앞에서 그랬듯, 한국어의 head-finality를 고려하여 문장 뒷쪽에서부터 padding하였습니다. 

---

The below is the evaluation phase utilizing the case of dense character embedding, which is the best case among the five feature engineerings above. Since the feature size is much bigger than the morpheme-based models, training epoch was extended to 50 for a fair comparison. The convergence was significantly slow compared with the morpheme-based cases, but we've obtained compatible accuracy (0.8802 > 0.8823) and F1 score (0.7906 > 0.7934) with respect to the performance with the morpheme-bilstm model. We assume that this result originates in the property of the syllables in Korean as subwords and the nature of the intention understanding task as a syntax-semantic task. 

```python
validate_bilstm(fci_rec,fci_label,32,128,class_weights_fci,0.1,16,'model/tutorial/rec_char_dense')
```

```properties
# CONSOLE RESULT
>>> validate_bilstm(fci_rec,fci_label,32,128,class_weights_fci,0.1,16,'model/tutorial/rec_char_dense')
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_12 (Bidirectio (None, 64)                34048     
_________________________________________________________________
dense_43 (Dense)             (None, 128)               8320      
_________________________________________________________________
dense_44 (Dense)             (None, 7)                 903       
=================================================================
Total params: 43,271
Trainable params: 43,271
Non-trainable params: 0
_________________________________________________________________
Train on 55129 samples, validate on 6126 samples
Epoch 1/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.6033 - acc: 0.7970— val_f1: 0.611853 — val_precision: 0.759945 — val_reca
— val_f1_w: 0.805877 — val_precision_w: 0.822164 — val_recall_w: 0.821907
55129/55129 [==============================] - 169s 3ms/step - loss: 0.6034 - acc: 0.7970 - val_loss: 0.5275 - val_acc: 0.8219
Epoch 2/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.4744 - acc: 0.8399— val_f1: 0.694396 — val_precision: 0.770346 — val_reca
— val_f1_w: 0.834075 — val_precision_w: 0.840706 — val_recall_w: 0.840679
55129/55129 [==============================] - 165s 3ms/step - loss: 0.4743 - acc: 0.8399 - val_loss: 0.4752 - val_acc: 0.8407
Epoch 3/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.4298 - acc: 0.8538— val_f1: 0.735305 — val_precision: 0.779485 — val_reca
— val_f1_w: 0.851305 — val_precision_w: 0.851509 — val_recall_w: 0.855534
55129/55129 [==============================] - 165s 3ms/step - loss: 0.4298 - acc: 0.8538 - val_loss: 0.4277 - val_acc: 0.8555
Epoch 4/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.4013 - acc: 0.8632— val_f1: 0.743459 — val_precision: 0.793688 — val_reca
— val_f1_w: 0.850187 — val_precision_w: 0.859941 — val_recall_w: 0.854228
55129/55129 [==============================] - 164s 3ms/step - loss: 0.4012 - acc: 0.8632 - val_loss: 0.4334 - val_acc: 0.8542
Epoch 5/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3808 - acc: 0.8699— val_f1: 0.759329 — val_precision: 0.791125 — val_reca
— val_f1_w: 0.862013 — val_precision_w: 0.863887 — val_recall_w: 0.864838
55129/55129 [==============================] - 165s 3ms/step - loss: 0.3809 - acc: 0.8699 - val_loss: 0.4008 - val_acc: 0.8648
Epoch 6/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.3639 - acc: 0.8744— val_f1: 0.772891 — val_precision: 0.818381 — val_reca
— val_f1_w: 0.867532 — val_precision_w: 0.868421 — val_recall_w: 0.870715
55129/55129 [==============================] - 165s 3ms/step - loss: 0.3638 - acc: 0.8744 - val_loss: 0.3888 - val_acc: 0.8707
Epoch 7/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3497 - acc: 0.8791— val_f1: 0.769384 — val_precision: 0.800740 — val_reca
— val_f1_w: 0.865818 — val_precision_w: 0.866474 — val_recall_w: 0.869083
55129/55129 [==============================] - 165s 3ms/step - loss: 0.3496 - acc: 0.8791 - val_loss: 0.3859 - val_acc: 0.8691
Epoch 8/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3363 - acc: 0.8844— val_f1: 0.776859 — val_precision: 0.809094 — val_reca
— val_f1_w: 0.871162 — val_precision_w: 0.870220 — val_recall_w: 0.874306
55129/55129 [==============================] - 165s 3ms/step - loss: 0.3363 - acc: 0.8844 - val_loss: 0.3775 - val_acc: 0.8743
Epoch 9/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3238 - acc: 0.8888— val_f1: 0.769087 — val_precision: 0.823962 — val_reca
— val_f1_w: 0.871311 — val_precision_w: 0.872008 — val_recall_w: 0.874633
55129/55129 [==============================] - 164s 3ms/step - loss: 0.3238 - acc: 0.8888 - val_loss: 0.3690 - val_acc: 0.8746
Epoch 10/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3147 - acc: 0.8912— val_f1: 0.778358 — val_precision: 0.815374 — val_reca
— val_f1_w: 0.871938 — val_precision_w: 0.872361 — val_recall_w: 0.873980
55129/55129 [==============================] - 164s 3ms/step - loss: 0.3148 - acc: 0.8912 - val_loss: 0.3691 - val_acc: 0.8740
Epoch 11/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3051 - acc: 0.8944— val_f1: 0.775244 — val_precision: 0.801135 — val_reca
— val_f1_w: 0.872543 — val_precision_w: 0.872058 — val_recall_w: 0.875449
55129/55129 [==============================] - 165s 3ms/step - loss: 0.3051 - acc: 0.8944 - val_loss: 0.3665 - val_acc: 0.8754
Epoch 12/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2970 - acc: 0.8981— val_f1: 0.781538 — val_precision: 0.809303 — val_reca
— val_f1_w: 0.875536 — val_precision_w: 0.875564 — val_recall_w: 0.878387
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2970 - acc: 0.8981 - val_loss: 0.3629 - val_acc: 0.8784
Epoch 13/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2891 - acc: 0.9001— val_f1: 0.781662 — val_precision: 0.795075 — val_reca
— val_f1_w: 0.872933 — val_precision_w: 0.873100 — val_recall_w: 0.873817
55129/55129 [==============================] - 164s 3ms/step - loss: 0.2891 - acc: 0.9001 - val_loss: 0.3684 - val_acc: 0.8738
Epoch 14/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2811 - acc: 0.9032— val_f1: 0.778117 — val_precision: 0.799786 — val_reca
— val_f1_w: 0.873586 — val_precision_w: 0.874717 — val_recall_w: 0.875775
55129/55129 [==============================] - 164s 3ms/step - loss: 0.2810 - acc: 0.9032 - val_loss: 0.3671 - val_acc: 0.8758
Epoch 15/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2744 - acc: 0.9035— val_f1: 0.784468 — val_precision: 0.810065 — val_reca
— val_f1_w: 0.874109 — val_precision_w: 0.873767 — val_recall_w: 0.876265
55129/55129 [==============================] - 164s 3ms/step - loss: 0.2744 - acc: 0.9036 - val_loss: 0.3618 - val_acc: 0.8763
Epoch 16/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2686 - acc: 0.9063— val_f1: 0.780115 — val_precision: 0.807169 — val_reca
— val_f1_w: 0.874687 — val_precision_w: 0.874115 — val_recall_w: 0.876755
55129/55129 [==============================] - 164s 3ms/step - loss: 0.2687 - acc: 0.9063 - val_loss: 0.3641 - val_acc: 0.8768
Epoch 17/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2615 - acc: 0.9094— val_f1: 0.782015 — val_precision: 0.796383 — val_reca
— val_f1_w: 0.874941 — val_precision_w: 0.876734 — val_recall_w: 0.875939
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2616 - acc: 0.9094 - val_loss: 0.3640 - val_acc: 0.8759
Epoch 18/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2548 - acc: 0.9099— val_f1: 0.790453 — val_precision: 0.812157 — val_reca
— val_f1_w: 0.880186 — val_precision_w: 0.880995 — val_recall_w: 0.881815
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2548 - acc: 0.9098 - val_loss: 0.3696 - val_acc: 0.8818
Epoch 19/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2484 - acc: 0.9136— val_f1: 0.787465 — val_precision: 0.791837 — val_reca
— val_f1_w: 0.877686 — val_precision_w: 0.880073 — val_recall_w: 0.877245
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2484 - acc: 0.9136 - val_loss: 0.3612 - val_acc: 0.8772
Epoch 20/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2432 - acc: 0.9150— val_f1: 0.764492 — val_precision: 0.800527 — val_reca
— val_f1_w: 0.863816 — val_precision_w: 0.865599 — val_recall_w: 0.867124
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2431 - acc: 0.9150 - val_loss: 0.4086 - val_acc: 0.8671
Epoch 21/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2364 - acc: 0.9179— val_f1: 0.786431 — val_precision: 0.825723 — val_reca
— val_f1_w: 0.877461 — val_precision_w: 0.877627 — val_recall_w: 0.880346
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2364 - acc: 0.9179 - val_loss: 0.3791 - val_acc: 0.8803
Epoch 22/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2314 - acc: 0.9198— val_f1: 0.784653 — val_precision: 0.813028 — val_reca
— val_f1_w: 0.879205 — val_precision_w: 0.880109 — val_recall_w: 0.880999
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2314 - acc: 0.9198 - val_loss: 0.3741 - val_acc: 0.8810
Epoch 23/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2251 - acc: 0.9218— val_f1: 0.780445 — val_precision: 0.804015 — val_reca
— val_f1_w: 0.875199 — val_precision_w: 0.874711 — val_recall_w: 0.877897
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2251 - acc: 0.9218 - val_loss: 0.3859 - val_acc: 0.8779
Epoch 24/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2195 - acc: 0.9240— val_f1: 0.784534 — val_precision: 0.789635 — val_reca
— val_f1_w: 0.876094 — val_precision_w: 0.875345 — val_recall_w: 0.877081
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2195 - acc: 0.9240 - val_loss: 0.3862 - val_acc: 0.8771
Epoch 25/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2145 - acc: 0.9252— val_f1: 0.789590 — val_precision: 0.800697 — val_reca
— val_f1_w: 0.879318 — val_precision_w: 0.878898 — val_recall_w: 0.880346
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2145 - acc: 0.9252 - val_loss: 0.3892 - val_acc: 0.8803
Epoch 26/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2087 - acc: 0.9281— val_f1: 0.786858 — val_precision: 0.804829 — val_reca
— val_f1_w: 0.878004 — val_precision_w: 0.877461 — val_recall_w: 0.879693
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2087 - acc: 0.9281 - val_loss: 0.3853 - val_acc: 0.8797
Epoch 27/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2042 - acc: 0.9290— val_f1: 0.779832 — val_precision: 0.808263 — val_reca
— val_f1_w: 0.873661 — val_precision_w: 0.872953 — val_recall_w: 0.876755
55129/55129 [==============================] - 165s 3ms/step - loss: 0.2042 - acc: 0.9290 - val_loss: 0.4095 - val_acc: 0.8768
Epoch 28/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.1993 - acc: 0.9306— val_f1: 0.792366 — val_precision: 0.799918 — val_reca
— val_f1_w: 0.881174 — val_precision_w: 0.881553 — val_recall_w: 0.881489
55129/55129 [==============================] - 165s 3ms/step - loss: 0.1994 - acc: 0.9306 - val_loss: 0.3898 - val_acc: 0.8815
Epoch 29/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.1939 - acc: 0.9330— val_f1: 0.787657 — val_precision: 0.799441 — val_reca
— val_f1_w: 0.878101 — val_precision_w: 0.877934 — val_recall_w: 0.879367
55129/55129 [==============================] - 165s 3ms/step - loss: 0.1939 - acc: 0.9330 - val_loss: 0.4101 - val_acc: 0.8794
Epoch 30/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.1895 - acc: 0.9333— val_f1: 0.767905 — val_precision: 0.779847 — val_reca
— val_f1_w: 0.863264 — val_precision_w: 0.863594 — val_recall_w: 0.865491
55129/55129 [==============================] - 165s 3ms/step - loss: 0.1896 - acc: 0.9332 - val_loss: 0.4535 - val_acc: 0.8655
Epoch 31/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.1856 - acc: 0.9350— val_f1: 0.787478 — val_precision: 0.806525 — val_reca
— val_f1_w: 0.875649 — val_precision_w: 0.874640 — val_recall_w: 0.877571
55129/55129 [==============================] - 165s 3ms/step - loss: 0.1856 - acc: 0.9350 - val_loss: 0.4093 - val_acc: 0.8776
Epoch 32/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.1798 - acc: 0.9380— val_f1: 0.796609 — val_precision: 0.802828 — val_reca
— val_f1_w: 0.877572 — val_precision_w: 0.877537 — val_recall_w: 0.878224
55129/55129 [==============================] - 165s 3ms/step - loss: 0.1799 - acc: 0.9380 - val_loss: 0.4327 - val_acc: 0.8782
Epoch 33/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.1760 - acc: 0.9389— val_f1: 0.794457 — val_precision: 0.807102 — val_reca
— val_f1_w: 0.878977 — val_precision_w: 0.879220 — val_recall_w: 0.879693
55129/55129 [==============================] - 164s 3ms/step - loss: 0.1760 - acc: 0.9389 - val_loss: 0.4271 - val_acc: 0.8797
Epoch 34/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.1714 - acc: 0.9403— val_f1: 0.794570 — val_precision: 0.800183 — val_reca
— val_f1_w: 0.874815 — val_precision_w: 0.875788 — val_recall_w: 0.874959
55129/55129 [==============================] - 165s 3ms/step - loss: 0.1713 - acc: 0.9404 - val_loss: 0.4511 - val_acc: 0.8750
Epoch 35/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.1675 - acc: 0.9414— val_f1: 0.793498 — val_precision: 0.804919 — val_reca
— val_f1_w: 0.881061 — val_precision_w: 0.881226 — val_recall_w: 0.882305
55129/55129 [==============================] - 165s 3ms/step - loss: 0.1675 - acc: 0.9414 - val_loss: 0.4274 - val_acc: 0.8823
```

* 이제 evaluation part입니다. 그런데 다섯 개나 되는 모델의 train 결과를 모두 올리는 건 좀 무리수인 것 같네요. 그래서 결론만 말씀드리면, one-hot encoded char < Cho < Shin < Song < dense char 의 순서로 성능이 좋습니다 ㅎㅎ 아무래도 궁금하실 dense char embedding의 트레이닝 phase만을 여기엔 업로드하도록 하겠습니다. 다른 알고리즘들은 validate_bilstm 모듈에 한번씩 돌려보면서 성능을 살펴보시면 좋을 것 같아요!

* 비록 feature size의 차이로 training epoch가 좀 늘어나긴 했지만 이는 convergence를 유도하기 위함이고, converge했다는 가정 하에서 morpheme based model보다 근소하게 더 성능이 좋음을 확인할 수 있었습니다. 이는 좀 의외의 결과였는데요, morpheme-based model과 다르게 여기서는 의미에 관련된 어떤 preprocessing도 거치지 않고 raw data만을 dictionary embedding한 것이기 때문입니다. 여기서, 한국어에서는 각 음절이 어느 정도 subword 역할을 해 주며 그 sequential한 배열로부터 문장의 의도를 파악하면 형태소 단위의 분석보다 때로는 더 정확할 수 있다는 것을 짐작할 수 있습니다. 좋은 성능의 배경에는 intention understanding 이라는 task의 특징이, 그리고 한국어의 agglutinative language로써의 특징이 있겠지만요.

---

Advanced from the vanilla models we've utilized so far, we may customize some Keras layers in the following chapter, to boost the performance.

* 지금까지 우리가 사용한 알고리즘들은 모두 vanilla cnn/bilstm이었고, 별도로 레이어를 쪼개고 합하고 곱하고 하지는 않았습니다. 하지만 back propagation을 이용한 트레이닝들의 장점은 연산의 커스터마이징이 가능하다는 점이며, 케라스에서도 그러한 연산들은 대부분 가능합니다. 다음 꼭지부터는 cnn과 bilstm을 엮어서 모델을 짜는 방법을 한번 알아보도록 하겠습니다.

## 9. Concatenation of CNN and RNN layers

Now, we get back to the morpheme-based approaches. Here, for the first time we customize the layers for a new implementation! The procedure is the most simple one, a concatenation of two separate networks. We've done with CNN and RNN(BiLSTM)-based approaches so far, thus, a concatenation of the two systems will be executed. Beyond simply putting *model.xxx*, we need some unseen module at this point, **Model**. This helps make in-out of the customized layers.

```python
from keras.models import Model
import keras.layers as layers
from keras.layers import Input
from keras.layers.core import Dense, Dropout
```

* 이제 다시 형태소 기반의 접근으로 돌아가 보도록 하겠습니다. 이제 처음으로, 단순히 model.머시기로 쌓는 게 아닌, 커스터마이즈드 레이어를 만들어보려고 해요! 물론 가장 기본적인, simple concatenation부터 시작해보도록 하겠습니다. 대상은 앞서 다루었던 CNN과 RNN(BiLSTM)입니다. 이를 위해서, 앞과는 다른 모듈들을 import할 필요가 있겠죠. 바로 **Model** 입니다. 커스터마이즈드 레이어의 입출력을 다룰 수 있게 해주죠!

---

The following is a simple code for a concatenation of CNN and RNN networks. The input dimensions were filled with the property regarding RNN input. There are some points you should look at: the dense layers after CNN and RNN summarization, the line with *layer.concatenate*, and the model declaration stage *model = Model(inputs = [cnn,rnn], outputs = [main_output])*. Dropouts were added since the model became so bigger than the previous ones. Wait ... there is another unseen function ... in callbacks?

```python
def validate_cnnrnn(conv,rnn,train_y,filters,hidden_lstm,hidden_dim,cw,filename):
    cnn_input = Input(shape=(len(rnn[0]),len(rnn[0][0]),1), dtype='float32')
    cnn_layer = layers.Conv2D(filters,(3,len(rnn[0][0])),activation='relu')(cnn_input)
    cnn_layer = layers.MaxPooling2D((2,1))(cnn_layer)
    cnn_layer = layers.Conv2D(filters,(3,1),activation='relu')(cnn_layer)
    cnn_layer = layers.Flatten()(cnn_layer)
    cnn_output= Dense(hidden_dim, activation='relu')(cnn_layer)
    rnn_input = Input(shape=(len(rnn[0]),len(rnn[0][0])), dtype='float32')
    rnn_layer = Bidirectional(LSTM(hidden_lstm))(rnn_input)
    rnn_output= Dense(hidden_dim, activation='relu')(rnn_layer)
    output    = layers.concatenate([cnn_output,rnn_output])
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    output    = Dense(hidden_dim, activation='relu')(output)
    output    = Dropout(0.3)(output)
    main_output = Dense(7,activation='softmax')(output)
    model = Sequential()
    model = Model(inputs=[cnn_input,rnn_input], outputs=[main_output])
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro_2input,checkpoint]
    model.summary()
    model.fit([conv,rnn],train_y,validation_split=0.1,epochs=30,batch_size=16,callbacks=callbacks_list,class_weight=cw)
```

상기 코드는 간단하게 CNN과 RNN 구조를 concatenate해 줍니다! 앞에서부터 쭉 보면 CNN_input과 CNN_output을 정의하고, RNN_input과 RNN_output을 정의하고, 두 레이어를 layer.concatenate를 이용해 합친 뒤, 앞서 했던 output에 대한 방법론과 비슷하게 진행하는 것을 파악할 수 있지요. 여기서 중요하게 볼 point들은 1. CNN과 RNN을 Dense layer로 summarize하여 concatenate하기 쉽게 만드는 부분, 2. *layer.concatenate*로 concatenate하는 부분, 3. *Model* 펑션으로 inputs와 outputs를 이은 하나의 큰 덩어리를 만드는 부분입니다. Dropout은 그냥 모델이 너무 커져서 넣어 보았습니다 ㅎㅎ 어... 그런데 callback부분에 못 보던 내용이 있군요?

---

As in Keras, we need to define another callback function for F1 score if we adopt more than one input. Actually, a different format of function is required for every number of more-than-one-input! Very annoying, but anyway, we've adopted two inputs, thus let's make up another function for the evaluation.

```python
class Metricsf1macro_2input(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
  self.val_f1s_w = []
  self.val_recalls_w = []
  self.val_precisions_w = []
 def on_epoch_end(self, epoch, logs={}):
  if len(self.validation_data)>2:
   val_predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[2]
  else:
   val_predict = np.asarray(self.model.predict(self.validation_data[0]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[1]
  _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
  _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
  _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
  _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
  _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
  _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  self.val_f1s_w.append(_val_f1_w)
  self.val_recalls_w.append(_val_recall_w)
  self.val_precisions_w.append(_val_precision_w)
  print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
  print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro_2input = Metricsf1macro_2input()
```

매-우 귀찮지만, 케라스에서 하나 이상의 input을 가진 모델에 대해 F1 score을 얻고 싶다면, 앞서 정의한 것과 다른, 별도의 function을 만들어야 합니다... 왜 이랬을까요? 아마 그쪽에서도 귀찮아서가 아닐까 싶습니다. 어쨌든 evaluation을 해야 하니, 정의를 해 보도록 하겠습니다.

---

Now, let's validate with the morpheme-based feaftures!

```python
validate_cnnrnn(fci_conv,fci_rec,fci_label,32,32,128,class_weights_fci,'model/modelfci/charcnn+bilstm')
```

We can see that the performance has degenerated even compared with the vanilla BiLSTM approach. Well, not always the bigger model results in the better performance. We infer that this originates in the distortion that comes from the difference in the way that CNN and RNN solve the problem. Next, we try another module, *attention network*, by introducing the backend of Keras.

```properties
# CONSOLE RESULT
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_7 (InputLayer)            (None, 30, 100, 1)   0                                            
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 28, 1, 32)    9632        input_7[0][0]                    
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 14, 1, 32)    0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 12, 1, 32)    3104        max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
input_8 (InputLayer)            (None, 30, 100)      0                                            
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 384)          0           conv2d_8[0][0]                   
__________________________________________________________________________________________________
bidirectional_3 (Bidirectional) (None, 64)           34048       input_8[0][0]                    
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 128)          49280       flatten_4[0][0]                  
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 128)          8320        bidirectional_3[0][0]            
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 256)          0           dense_11[0][0]                   
                                                                 dense_12[0][0]                   
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 128)          32896       concatenate_3[0][0]              
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 128)          0           dense_13[0][0]                   
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 128)          16512       dropout_5[0][0]                  
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 128)          0           dense_14[0][0]                   
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 7)            903         dropout_6[0][0]                  
==================================================================================================
Total params: 154,695
Trainable params: 154,695
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 55129 samples, validate on 6126 samples
Epoch 1/30
55104/55129 [============================>.] - ETA: 0s - loss: 0.6263 - acc: 0.7916— val_f1: 0.730286 — val_precision: 0.768703 — val_recall: 0.705825
— val_f1_w: 0.843637 — val_precision_w: 0.842989 — val_recall_w: 0.847698
55129/55129 [==============================] - 90s 2ms/step - loss: 0.6261 - acc: 0.7916 - val_loss: 0.4664 - val_acc: 0.8477
Epoch 2/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.4405 - acc: 0.8548— val_f1: 0.758616 — val_precision: 0.786346 — val_recall: 0.740990
— val_f1_w: 0.856354 — val_precision_w: 0.860634 — val_recall_w: 0.857982
55129/55129 [==============================] - 89s 2ms/step - loss: 0.4405 - acc: 0.8548 - val_loss: 0.4282 - val_acc: 0.8580
Epoch 3/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.3859 - acc: 0.8731— val_f1: 0.759942 — val_precision: 0.804874 — val_recall: 0.729082
— val_f1_w: 0.861656 — val_precision_w: 0.863499 — val_recall_w: 0.865002
55129/55129 [==============================] - 89s 2ms/step - loss: 0.3859 - acc: 0.8731 - val_loss: 0.3992 - val_acc: 0.8650
Epoch 4/30
55104/55129 [============================>.] - ETA: 0s - loss: 0.3492 - acc: 0.8831— val_f1: 0.781057 — val_precision: 0.795622 — val_recall: 0.769233
— val_f1_w: 0.868542 — val_precision_w: 0.869711 — val_recall_w: 0.869899
55129/55129 [==============================] - 89s 2ms/step - loss: 0.3493 - acc: 0.8831 - val_loss: 0.3879 - val_acc: 0.8699
Epoch 5/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.3177 - acc: 0.8927— val_f1: 0.779071 — val_precision: 0.829605 — val_recall: 0.746311
— val_f1_w: 0.865831 — val_precision_w: 0.870711 — val_recall_w: 0.868919
55129/55129 [==============================] - 90s 2ms/step - loss: 0.3177 - acc: 0.8927 - val_loss: 0.4028 - val_acc: 0.8689
Epoch 6/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.2919 - acc: 0.9011— val_f1: 0.783441 — val_precision: 0.804650 — val_recall: 0.768078
— val_f1_w: 0.872711 — val_precision_w: 0.873444 — val_recall_w: 0.873980
55129/55129 [==============================] - 89s 2ms/step - loss: 0.2919 - acc: 0.9011 - val_loss: 0.3968 - val_acc: 0.8740
Epoch 7/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.2667 - acc: 0.9091— val_f1: 0.784232 — val_precision: 0.819105 — val_recall: 0.761072
— val_f1_w: 0.870747 — val_precision_w: 0.870663 — val_recall_w: 0.873164
55129/55129 [==============================] - 90s 2ms/step - loss: 0.2667 - acc: 0.9091 - val_loss: 0.4189 - val_acc: 0.8732
Epoch 8/30
55104/55129 [============================>.] - ETA: 0s - loss: 0.2447 - acc: 0.9174— val_f1: 0.781662 — val_precision: 0.811667 — val_recall: 0.758755
— val_f1_w: 0.868505 — val_precision_w: 0.869532 — val_recall_w: 0.871205
55129/55129 [==============================] - 89s 2ms/step - loss: 0.2447 - acc: 0.9174 - val_loss: 0.4401 - val_acc: 0.8712
Epoch 9/30
55104/55129 [============================>.] - ETA: 0s - loss: 0.2226 - acc: 0.9235— val_f1: 0.777220 — val_precision: 0.781946 — val_recall: 0.782025
— val_f1_w: 0.873209 — val_precision_w: 0.874389 — val_recall_w: 0.873980
55129/55129 [==============================] - 90s 2ms/step - loss: 0.2227 - acc: 0.9235 - val_loss: 0.4419 - val_acc: 0.8740
Epoch 10/30
55104/55129 [============================>.] - ETA: 0s - loss: 0.2055 - acc: 0.9280— val_f1: 0.776107 — val_precision: 0.795161 — val_recall: 0.763229
— val_f1_w: 0.867720 — val_precision_w: 0.867430 — val_recall_w: 0.869246
55129/55129 [==============================] - 90s 2ms/step - loss: 0.2055 - acc: 0.9280 - val_loss: 0.4681 - val_acc: 0.8692
Epoch 11/30
55104/55129 [============================>.] - ETA: 0s - loss: 0.1893 - acc: 0.9344— val_f1: 0.788884 — val_precision: 0.796599 — val_recall: 0.782946
— val_f1_w: 0.875588 — val_precision_w: 0.876054 — val_recall_w: 0.875612
55129/55129 [==============================] - 89s 2ms/step - loss: 0.1892 - acc: 0.9345 - val_loss: 0.4614 - val_acc: 0.8756
Epoch 12/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.1749 - acc: 0.9389— val_f1: 0.784650 — val_precision: 0.809236 — val_recall: 0.767474
— val_f1_w: 0.871932 — val_precision_w: 0.871065 — val_recall_w: 0.874306
55129/55129 [==============================] - 89s 2ms/step - loss: 0.1750 - acc: 0.9389 - val_loss: 0.4874 - val_acc: 0.8743
Epoch 13/30
55120/55129 [============================>.] - ETA: 0s - loss: 0.1613 - acc: 0.9439— val_f1: 0.790170 — val_precision: 0.813151 — val_recall: 0.771651
— val_f1_w: 0.874813 — val_precision_w: 0.874452 — val_recall_w: 0.876755
55129/55129 [==============================] - 90s 2ms/step - loss: 0.1613 - acc: 0.9439 - val_loss: 0.5269 - val_acc: 0.8768
```

어... vanilla BiLSTM보다 성능이 떨어졌네요 ㅎㅎ 뭐 그럴수도 있죠... 아무래도 CNN과 BiLSTM이 infer하는 과정에서 양상이 달라, jointly training에서는 혼선이 생긴 것이 아닌가 싶습니다. model을 merge하는 것이 꼭 좋은 결과를 가져오지는 않는 것 같아요. 다음 장에서는 backend인 tensorflow를 끌어 와서 self-attentive BiLSTM을 구현해 보도록 하겠습니다.

## 10. Self-attentive BiLSTM

The second to the final step is applying 'the' attention mechanism, which has shifted paradigm of deep learning architectures, and still shifting... Many will be familiar with the [attention model](https://arxiv.org/abs/1409.0473) which came out along with [RNN encoder-decoder](https://arxiv.org/abs/1406.1078), or the self-attention which was suggested in [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need), which deals with seq2seq-style problems such as machine translation. Though the inherit philosophy may be consistent, here, we introduce a [structured self attentive embedding](https://arxiv.org/abs/1703.03130) which was suggested for an effective sentence classification.

The key idea in this paper is to train an additional attention layer that assigns weight to each hidden layer of the BiLSTM structure. For this, a context vector of the size same as the hidden layer width, i.e. 64 as will be implemented, is separately defined and jointly trained, in the manner that **it is column-wisely multiplied with the MLP from each hidden layer to yield the attention vector**. The attention vector is recursively multiplied to the hidden layers so that the weighted sum becomes the final summarization for the fine-tuning. The code is implemented below; with the new function *lambda* which enables us to customize the layers in somewhat sophiscated ways. We also need to import the backend of Keras, here TensorFlow, for the layer-level and specific operations.

```python
from keras.layers import Lambda
import keras.backend as K

def validate_rnn_self_drop(x_rnn,x_y,hidden_lstm,hidden_con,hidden_dim,cw,val_sp,bat_size,filename):
    char_r_input = Input(shape=(len(x_rnn[0]),len(x_rnn[0][0])),dtype='float32')
    r_seq = Bidirectional(LSTM(hidden_lstm,return_sequences=True))(char_r_input)
    r_att = Dense(hidden_con, activation='tanh')(r_seq)
    att_source   = np.zeros((len(x_rnn),hidden_con))
    att_test     = np.zeros((len(x_rnn),hidden_con))
    att_input    = Input(shape=(hidden_con,), dtype='float32')
    att_vec      = Dense(hidden_con,activation='relu')(att_input)
    att_vec      = Dropout(0.3)(att_vec)
    att_vec      = Dense(hidden_con,activation='relu')(att_vec)
    att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([att_vec,r_att])
    att_vec = Dense(len(x_rnn[0]),activation='softmax')(att_vec)
    att_vec = layers.Reshape((len(x_rnn[0]),1))(att_vec)
    r_seq   = layers.multiply([att_vec,r_seq])
    r_seq   = Lambda(lambda x: K.sum(x, axis=1))(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    main_output = Dense(int(max(x_y)+1),activation='softmax')(r_seq)
    model = Sequential()
    model = Model(inputs=[char_r_input,att_input],outputs=[main_output])
    model.summary()
    model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro_2input,checkpoint]
    model.fit([x_rnn,att_source],x_y,validation_split=val_sp,epochs=50,batch_size= bat_size ,callbacks=callbacks_list,class_weight=cw)
```

At the very beginning of the code, BiLSTM module is defined so that each hidden layer can be fed as an input of MLP (here a single layer was utilized though) whose final size is the same as *the context vector*, 64 (*hidden_con*), to make up *r_att*. Next, from an attention source *zeros*, an attention vector *att_vec* is yielded by MLP, with the size of *hidden_con*, and is multiplied column-wisely to *r_att*! This finally makes up an attention vector that is recursively multiplied to the hidden layer sequence to yield the weighted sum. The rest are the same, but due to the model being large, we raised the number of epochs to 50. The same callback functions were utilized as in the last chapter since we utilized two inputs, RNN dataset and attention source (zeros).

<p align="center">
    <image src="https://github.com/warnikchow/dlk2nlp/blob/master/image/selfAA.png" width="700"><br/>
    (image from Lin 2017)
	 	 
요즘, 아니 꽤 오랫동안 NLP와 ML에서 혁신을 가져왔던 attention mechanism을 이제서야 만나볼 수 있게 되었습니다. 많은 분들이 익숙하신 내용은 주로 machine translation과 함께 사용되었던 [attention model](https://arxiv.org/abs/1409.0473)이나 Transformer의 [self attention](http://papers.nips.cc/paper/7181-attention-is-all-you-need)일 텐데요, 여기서는 비슷한 시기에 sentence classification task를 위해 등장한 structured self-attentive embedding을 살펴보도록 하겠습니다. self attention이 나온다기는 좀 뭐하지만, attention vector을 활용하여 sentence classification에 사용되는 latent variable들에 정보를 주는데, 그 source가 자기 자신이라는 점이 self-attention과 유사한 측면이 있다고 생각됩니다.

아주 러프하게 말하면, 전체 procedure는 BiLSTM의 hidden layer sequence에 column-wise하게 곱해지는 attention vector를 얻는 데에 있어, context vector라는 개념을 도입하는 데에 있습니다. Context vector의 dimension을 *hidden_con*이라고 한다면, 우선 BiLSTM에서 MLP를 통해 *hidden_con*의 크기를 가진 dense layer을 각 hidden layer에 대해 뽑아낸 후, context vector와 column-wise한 inner product를 통해 attention vector을 얻는 것이죠. 어떻게 말해도 복잡하네요... 위에 있는 그림을 보는 것이 좀 더 이해가 빠를 것입니다 ㅎㅎ 여튼 여기서 중요한 점은, 케라스를 이용해서도 이 과정을 구현할 수 있다, 그런데 구현하려면 backend와 lambda라는 녀석들이 필요하다! 이 정도인 것 같네요.

---

The implementation might not be optimum, but well, we followed the description in the paper! The result shows that the self-attentive BiLSTM outperforms the vanilla one and the CNN-RNN concatenation, and matches with the character-based approach (which incorporates pretrained word embedding)! But it's not sure if we can push through the wall of 90% ...

```properties
>>> validate_rnn_self_drop(fci_rec,fci_label,32,64,256,class_weights_fci,0.1,16,'model/tutorial/rec_self_drop')

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_12 (InputLayer)           (None, 64)           0                                            
__________________________________________________________________________________________________
dense_20 (Dense)                (None, 64)           4160        input_12[0][0]                   
__________________________________________________________________________________________________
input_11 (InputLayer)           (None, 30, 100)      0                                            
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 64)           0           dense_20[0][0]                   
__________________________________________________________________________________________________
bidirectional_5 (Bidirectional) (None, 30, 64)       34048       input_11[0][0]                   
__________________________________________________________________________________________________
dense_21 (Dense)                (None, 64)           4160        dropout_8[0][0]                  
__________________________________________________________________________________________________
dense_19 (Dense)                (None, 30, 64)       4160        bidirectional_5[0][0]            
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 30)           0           dense_21[0][0]                   
                                                                 dense_19[0][0]                   
__________________________________________________________________________________________________
dense_22 (Dense)                (None, 30)           930         lambda_1[0][0]                   
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 30, 1)        0           dense_22[0][0]                   
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 30, 64)       0           reshape_1[0][0]                  
                                                                 bidirectional_5[0][0]            
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 64)           0           multiply_1[0][0]                 
__________________________________________________________________________________________________
dense_23 (Dense)                (None, 256)          16640       lambda_2[0][0]                   
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 256)          0           dense_23[0][0]                   
__________________________________________________________________________________________________
dense_24 (Dense)                (None, 256)          65792       dropout_9[0][0]                  
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 256)          0           dense_24[0][0]                   
__________________________________________________________________________________________________
dense_25 (Dense)                (None, 7)            1799        dropout_10[0][0]                 
==================================================================================================
Total params: 131,689
Trainable params: 131,689
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 55129 samples, validate on 6126 samples
Epoch 1/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.6279 - acc: 0.7886— val_f1: 0.711039 — val_precision: 0.807089 — val_recall: 0.670628
— val_f1_w: 0.831240 — val_precision_w: 0.842004 — val_recall_w: 0.839700
55129/55129 [==============================] - 86s 2ms/step - loss: 0.6279 - acc: 0.7886 - val_loss: 0.4957 - val_acc: 0.8397
Epoch 2/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.4680 - acc: 0.8442— val_f1: 0.754434 — val_precision: 0.791817 — val_recall: 0.731321
— val_f1_w: 0.848612 — val_precision_w: 0.848852 — val_recall_w: 0.851126
55129/55129 [==============================] - 88s 2ms/step - loss: 0.4680 - acc: 0.8442 - val_loss: 0.4490 - val_acc: 0.8511
Epoch 3/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.4244 - acc: 0.8578— val_f1: 0.771808 — val_precision: 0.812655 — val_recall: 0.743364
— val_f1_w: 0.858047 — val_precision_w: 0.858865 — val_recall_w: 0.861410
55129/55129 [==============================] - 86s 2ms/step - loss: 0.4243 - acc: 0.8578 - val_loss: 0.4134 - val_acc: 0.8614
Epoch 4/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.3948 - acc: 0.8673— val_f1: 0.751910 — val_precision: 0.806796 — val_recall: 0.719961
— val_f1_w: 0.856063 — val_precision_w: 0.856820 — val_recall_w: 0.860431
55129/55129 [==============================] - 87s 2ms/step - loss: 0.3948 - acc: 0.8673 - val_loss: 0.4073 - val_acc: 0.8604
Epoch 5/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3688 - acc: 0.8747— val_f1: 0.772794 — val_precision: 0.823605 — val_recall: 0.740154
— val_f1_w: 0.860423 — val_precision_w: 0.866368 — val_recall_w: 0.863206
55129/55129 [==============================] - 91s 2ms/step - loss: 0.3688 - acc: 0.8747 - val_loss: 0.4034 - val_acc: 0.8632
Epoch 6/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3482 - acc: 0.8801— val_f1: 0.782679 — val_precision: 0.834986 — val_recall: 0.751268
— val_f1_w: 0.867117 — val_precision_w: 0.871998 — val_recall_w: 0.871041
55129/55129 [==============================] - 89s 2ms/step - loss: 0.3482 - acc: 0.8801 - val_loss: 0.3943 - val_acc: 0.8710
Epoch 7/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.3285 - acc: 0.8869— val_f1: 0.775507 — val_precision: 0.834404 — val_recall: 0.740948
— val_f1_w: 0.866852 — val_precision_w: 0.870674 — val_recall_w: 0.871205
55129/55129 [==============================] - 86s 2ms/step - loss: 0.3284 - acc: 0.8869 - val_loss: 0.4053 - val_acc: 0.8712
Epoch 8/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.3166 - acc: 0.8914— val_f1: 0.789958 — val_precision: 0.832346 — val_recall: 0.761638
— val_f1_w: 0.874285 — val_precision_w: 0.877032 — val_recall_w: 0.877408
55129/55129 [==============================] - 90s 2ms/step - loss: 0.3166 - acc: 0.8914 - val_loss: 0.3843 - val_acc: 0.8774
Epoch 9/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.3011 - acc: 0.8956— val_f1: 0.796802 — val_precision: 0.811898 — val_recall: 0.786651
— val_f1_w: 0.872073 — val_precision_w: 0.872473 — val_recall_w: 0.873653
55129/55129 [==============================] - 91s 2ms/step - loss: 0.3011 - acc: 0.8956 - val_loss: 0.3862 - val_acc: 0.8737
Epoch 10/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2873 - acc: 0.9008— val_f1: 0.786752 — val_precision: 0.821892 — val_recall: 0.765839
— val_f1_w: 0.874281 — val_precision_w: 0.877892 — val_recall_w: 0.877408
55129/55129 [==============================] - 93s 2ms/step - loss: 0.2873 - acc: 0.9008 - val_loss: 0.3845 - val_acc: 0.8774
Epoch 11/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2755 - acc: 0.9048— val_f1: 0.796917 — val_precision: 0.856046 — val_recall: 0.760149
— val_f1_w: 0.877631 — val_precision_w: 0.881019 — val_recall_w: 0.881815
55129/55129 [==============================] - 92s 2ms/step - loss: 0.2755 - acc: 0.9048 - val_loss: 0.3812 - val_acc: 0.8818
Epoch 12/50
55088/55129 [============================>.] - ETA: 0s - loss: 0.2633 - acc: 0.9076— val_f1: 0.793011 — val_precision: 0.825051 — val_recall: 0.769426
— val_f1_w: 0.878768 — val_precision_w: 0.878530 — val_recall_w: 0.881325
55129/55129 [==============================] - 90s 2ms/step - loss: 0.2633 - acc: 0.9076 - val_loss: 0.3828 - val_acc: 0.8813
Epoch 13/50
55104/55129 [============================>.] - ETA: 0s - loss: 0.2515 - acc: 0.9121— val_f1: 0.791692 — val_precision: 0.792037 — val_recall: 0.792989
— val_f1_w: 0.873094 — val_precision_w: 0.872973 — val_recall_w: 0.873980
55129/55129 [==============================] - 91s 2ms/step - loss: 0.2515 - acc: 0.9122 - val_loss: 0.3975 - val_acc: 0.8740
Epoch 14/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2413 - acc: 0.9150— val_f1: 0.799131 — val_precision: 0.802153 — val_recall: 0.797500
— val_f1_w: 0.878161 — val_precision_w: 0.877972 — val_recall_w: 0.878877
55129/55129 [==============================] - 94s 2ms/step - loss: 0.2413 - acc: 0.9150 - val_loss: 0.3853 - val_acc: 0.8789
Epoch 15/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2295 - acc: 0.9190— val_f1: 0.800154 — val_precision: 0.813464 — val_recall: 0.789562
— val_f1_w: 0.876815 — val_precision_w: 0.876335 — val_recall_w: 0.877897
55129/55129 [==============================] - 96s 2ms/step - loss: 0.2294 - acc: 0.9190 - val_loss: 0.3951 - val_acc: 0.8779
Epoch 16/50
55120/55129 [============================>.] - ETA: 0s - loss: 0.2200 - acc: 0.9227— val_f1: 0.791979 — val_precision: 0.816055 — val_recall: 0.775612
— val_f1_w: 0.879622 — val_precision_w: 0.879747 — val_recall_w: 0.881978
55129/55129 [==============================] - 97s 2ms/step - loss: 0.2200 - acc: 0.9227 - val_loss: 0.4043 - val_acc: 0.8820
```

어찌어찌하여 character-level embedding과 비슷한 결과가 나오긴 했네요 ㅎㅎ 힘들었습니다...만 여기서 그나마 얻은 것은 케라스에서도 이런 layer 단위의 빡센 아키텍쳐 수립이 가능하다는 점이었네요 ㅎㅎ 이게 베스트 구현인지는 모르겠습니다만 어쨌든 논문에서 하라는 대로 다 구현도 했고 성능도 아까에 비해서는 올랐습니다! ㅠㅠ 그런데 과연 우린 90%의 벽을 넘을 수 있을까요? 그 전에 89% 벽을 넘을 수 있을까요..? 지금까지의 feature-based approach들과 차별화된 뭔가 다른 접근방법이 필요한 것은 아닐까요....?

## 11. Transformer, BERT, and after

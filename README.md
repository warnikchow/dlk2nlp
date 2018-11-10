# DLK2NLP
### Day-by-day Line-by-line Keras-based Korean NLP
## Sentence classification: From data construction to BiLSTM self-attention
* 문장 분류 task를 중심으로 본 한국어 NLP 튜토리얼입니다. 본 튜토리얼에 사용된 컨텐츠 일부는 저자가 운영하는 [페이스북 페이지](https://www.facebook.com/nobodybelongs/notes/)에 기고한 글에서 발췌하였음을 명시합니다. 

## Contents (to be updated)
[0. Corpus labelling](https://github.com/warnikchow/dlk2nlp/blob/master/README.md#0-corpus-labeling)</br>
[1. Data preprocessing](https://github.com/warnikchow/dlk2nlp#1-data-preprocessing)</br>
[2. One-hot encoding and sentence vector](https://github.com/warnikchow/dlk2nlp#2-one-hot-encoding-and-sentence-vector)</br>
[3. TF-IDF and basic classifiers](https://github.com/warnikchow/dlk2nlp#3-tf-idf-and-basic-classifiers)</br>
[4. Dense word embedding and document vectors](https://github.com/warnikchow/dlk2nlp#4-dense-word-embedding-and-document-vectors)</br>
[5. NN classifier using Keras](https://github.com/warnikchow/dlk2nlp#5-nn-classifier-using-keras)</br>
[6. CNN-based sentence classification](https://github.com/warnikchow/dlk2nlp#6-cnn-based-sentence-classification)</br>
[7. RNN (BiLSTM)-based sentence classification](https://github.com/warnikchow/dlk2nlp#7-rnn-bilstm-based-sentence-classification)</br>
[8. Character embedding](https://github.com/warnikchow/dlk2nlp#8-character-embedding)</br>
[9. Concatenation of CNN and RNN layers](https://github.com/warnikchow/dlk2nlp#9-concatenation-of-cnn-and-rnn-layers)</br>
[10. BiLSTM Self-attention](https://github.com/warnikchow/dlk2nlp#10-bilstm-self-attention)

## 0. Corpus labeling
The most annoying and confusing process.
Annotation guideline should be provided to annotators and more than two natives should be engaged in to make the labeling reliable and also for a computation of inter-annotator agreement (IAA). In this project, multi-class (7) annotation of short Korean utterances is utilized.
* 데이터를 만드는, 가장 귀찮은 과정입니다. 언어학적 직관은 1인분이기 때문에, 레이블링이 설득력을 얻기 위해서는 적어도 3명 이상의 1화자를 통한 레이블링으로 그 타당성을 검증해야 합니다 (아카데믹하게는...) 
* 본 프로젝트에서는 7-class의 한국어 문장들이 분류에 사용됩니다.

The task is on classification; especially about extracting intention from a single utterance with the punctuation removed, which is suggested in [3i4k](https://github.com/warnikchow/3i4k). As the description displays, the corpus was partially hand-annotated and incorporates the utterances which are generated or semi-automatically collected. Total number of utterances reaches 57K, with each label denoting</br>
**0: Fragments**</br>
**1: Statement**</br>
**2: Question**</br>
**3: Command**</br>
**4: Rhetorical question**</br>
**5: Rhetorical command**</br>
**6: Intonation-dependent utterance**</br>
where the IAA was computed 0.85 (quite high!) for the manually annotated 2K utterance set.
* 태스크는 의도 분류로써, [3i4k](https://github.com/warnikchow/3i4k) 프로젝트를 위해 제작된 DB를 사용합니다. 사실 국책과제에 쓰려고 만든건데 어차피 논문으로도 submit했으니 공개는 상관 없지 않을까 싶어요. 5만 7천 문장쯤으로 아주 규모가 크지는 않지만, 일단 수작업으로 2만 문장 정도에서 0.85의 IAA를 얻었으며 (꽤 높은 agreement!), 4만 문장 가량이 더 수집/생성되어 그래도 어느정도 쓸만한 데이터셋이 만들어졌습니다. 
* 레이블 7개는 위에 써 둔 것처럼, Statement~Rhetorical question까지의 clear한 의도 5가지와 (논문에선 clear-cut cases라고 칭했습니다만), 의도가 불분명한 Fragment (명사, 명사구, 혹은 불완전한 문장), 마지막으로 Intonation-dependent utterances *억양에 따라 의도가 달라지는 문형* 입니다. 마지막 레이블은 저 논문에서 하나의 레이블로 하기로 제안한 것이지만, 한국어 화자라면 어떤 문장들이 그런 성질을 가지는지 감이 올 것입니다. "뭐 먹고 싶어" "천천히 가고 있어" 같은 문장들이 그러한 유형이죠. Spoken language understanding에 아주 골머리를 썩이는 녀석들이기 때문에 따로 분류하기로 하였습니다. 
* Annotation guideline이 어떤 형식인지 궁금하신 분들은 [이곳](https://drive.google.com/open?id=1AvxzEHr7wccMw7LYh0J3Xbx5GLFfcvMW)을 참고하시면 됩니다.

## 1. Data preprocessing
For the next step, we should pay attention to how we can manage HANGUL, the letters of Korean writing system (WS). Korean WS, letter and its alphabets *Jamo* which was (maybe solely) invented by The Great King Sejong, incorporate special morpho-syllabic blocks which have a role of syllable and consist of CV(C), making Korean as a representative language with featural WS. The blocks are agglutinated to make up the word *eojeol*, which should actually be decomposed into morphemes for a semantically meaningful language processing. The spacings go between the *eojeal*s, to enhance the readability of a sentence. Many morphological analyzers give an additional spacing between the morphemes; some analyzers such as [Twitter](https://github.com/twitter/twitter-korean-text) simply give spaces, and there are the ones which conduct an elaborate tokenizing process of block decomposition, such as [Kkma](http://kkma.snu.ac.kr/). Total five analyzers are wrapped in the famous Korean natural language processing toolkit, [KoNLPy](http://konlpy.org/en/v0.4.4/). In this tutorial, we proceed with the Twitter analyzer, for its speed and to prevent the letters being decomposed.
* 한국어와 한글, 문자 체계, 형태소, 띄어쓰기 및 형태소 분석기에 대해 주절주절 설명해 봤는데요, 아마 한국어 L1 화자라면 대부분 익숙한 내용이실 테니 이 부분의 한글 설명은 생략하도록 하겠습니다 ㅎㅎ 
* 아직 국제 무대에서는 크게 주목받지는 못하지만 그래도 한국어는 화자 수로만 봐도 세계 15위권 안에 들고 진짜로 몇 안되는 featural WS (갓종갓왕님 1인 프로젝트 덕분에 sign writing이나 소설을 위해 만들어진 언어 등과 같은 문자형식 지위를 획득 ...) 이며 promissive와 같은 독특한 particle 덕분에 언어학 서적에서도 왕왕 나오는 언어 및 문자체계를 갖고 있습니다. 또한 BTS의 떡상으로 전세계 사람들이 한국과 한국어와 한글을 알아서 Korean NLP가 국제무대에서도 Chinese나 Arabic만큼 중요하게 다뤄지는 때가 오기를... 혹은 통일을 기원하며 떡상을 기다리는... 
* 여튼 여기서 한 얘기는 한국어 NLP에서 semantic하게 의미있는 작업을 위해서는 agglutinate된 block들을 morpheme 단위로 적절히 쪼개는 과정이 필수적이라는 얘기를 하고 있습니다. 저는 KoNLPy에 들어 있는 Twitter가 가볍고 자소분해를 하지 않아 많이 사용하는데, 여기서도 해당 모듈을 사용하도록 하겠습니다.

The most important part of Korean NLP lies in using Python 3.x. For the lower version, the encoding issue will bother you. That is, in Python 3.x, you don't have to perform an additional encoding for HANGUL to be read and written. [THIS](https://github.com/warnikchow/3i4k/blob/master/data/fci.txt) is the file url, which is a single *.txt* file that has the first column of the labels and the second column of the sentences. The dataset is split into train and test set of ratio 9:1 and the test set denotes last 10% of the corpus. The following code reads the dataset into the sentences and the labels, where the dataset is placed in the path 'data/fci.txt' for the directory you're running the console.

<pre><code>def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data
    
fci = read_data('data/fci.txt')
fci_data = [t[1] for t in fci]
fci_label= [int(t[0]) for t in fci]</code></pre>

* 한국어엔 역시 Python 3.x 죠.... 2.x로 한국어 인코딩은 지옥입니다 ㅠㅠ 처음에 아주 고생했네요. 어쨌든 제일 첫 삽은 data를 로딩하고 sentence와 label들로 나누는 과정입니다. 저는 [Lucy Park님의 유명한 슬라이드](https://www.lucypark.kr/docs/2015-pyconkr.pdf)에서 처음 봤지만, 외국 NLP 튜토리얼들에도 비슷하게 나와 있습니다. read_data로 파일을 읽어들이고, 탭(\t)으로 split한 후, data와 label array로 나누게 됩니다.

The last part of data preprocessing is tokenizing the sentence into morphemes, as emphasized previously. Although many character-based (morpho-syllabic blocks) or alphabet-based (consonants and vowels, or *Jamo*) approaches are utilized these days, morpheme-based approach is still meaningful due to the nature of Korean as an agglutinative language. For the sparse vector classification such as one-hot encoding and TF-IDF which will be displayed in the following chapter, we will adopt the morpheme sequence which can be obtained by the Twitter tokenizer.

* 데이터를 읽어들였으니, 이제 형태소 분리를 통해 어절이라는 큰 뭉텅이들을 좀 더 세밀한 단위로 나눠 보도록 하겠습니다. 음절 기반 방법, 혹은 자소 분해 방법들도 요즘 많이 사용되지만, 형태소 기반의 문장 표현이 아무래도 교착어인 한국어에서 가장 기본적인 접근 방법이 아닌가 싶습니다. 
* 아까 말씀드렸듯 트위터 형태소 분석기를 사용하여 어절들을 분리할 계획이며, 이는 one-hot encoding이나 TF-IDF를 이용한 sparse vector classification에 활용하기 위함입니다.

## 2. One-hot encoding and sentence vector
The fundamental of computational linguistics lies in making machines understand human utterances (or, natural language). Since it is difficult for even human beings to understand the **real meaning**, the first step is to represent the words and sentences into the numerics that are computable for the machine learning systems. Given the dictionary of vocabulary size N, the most famous and intuitive approach is one-hot encoding that assigns 1 for only one entry if a specific word is given.

* 전산 언어학은 기본적으로 컴퓨터한테 사람이 말하는 것, 즉 자연어를 이해시키는 과정이라고 할 수 있겠습니다. 그래서 우리는 컴퓨터로 하여금, '아 진짜 우리 말을 이해하진 못해도, 대충 어떤 내용인지 판단은 할 수 있게 해 보자' 생각을 하게 됩니다. 그렇게 해서 나오게 된 표현법 (representation) 중 가장 기본적으로 사용되고, 매우 강력한 편이라 아직까지도 많은 곳에서 사용하고 있는 방법론은 바로 단어의 1-hot encoding입니다. 
* 이 방법을 쓰기 위해선 '우리가 생각할 단어 전체 set = Dict'을 생각해야 합니다. 가장 먼저 생각해볼 수 있는 건 우리가 사용할 코퍼스에 있는 모든 단어들의 모임이죠. 코퍼스의 사이즈가 커질수록 Dict도 커질 것이고, 물론 단어는 countably infinite하게 만들어낼 수 있지만 여기선 '그나마 자주 쓰이는 녀석들'로 생각합시다. 
* 이 Dict가 생각보다 커서 실제로는 보통 많이 쓰이는 n만 단어 같이 제한을 둬서 잡습니다. 목적에 따라 functional한 녀석들은 아예 카운트하지 않기도 해요. 이 때 size(Dict) = N 이라 하면, 우리는 (고려하기로 한) 모든 단어들을 N-dim의 1-hot vector로 표현할 수 있게 되는 겁니다.

What should we do with these large-dimensional vectors? The first thing we can think of is a feature called *Bag-of-Words* (BoW). The literal meaning is a bag which contains words, and for Korean, it might be either word (*eojeol*), morpheme, characters, or alphabets (*Jamo*). Since we've decided only to use morphemes for the sparse representations, but these can be replaced with whatever feature you want to adopt. BoW approach is very simple; it just assigns 1 to the entry that corresponds with a word, 

* 이걸 갖고 뭘 하느냐? 가장 먼저 생각해볼 수 있는 것은 bag-of-words란 녀석입니다. 말 그대로 '단어가 든 가방' 이에요. 이 때 가방 = 문장 입니다. 문장 안에 어떤 단어들이 들었냐를 N-dim 1-hot vector들의 OR-sum operation (하나만 있어도 1됨) 으로 표현하는 거죠. 
* 예컨대 "신이 그댈 사랑해"라는 문장이 있고 '신이' = \[1 0 0 0 0 0 \], '그댈' = \[0 0 1 0 0 0\], '사랑해' = \[0 0 0 0 0 1\]로 표현된다고 합시다. 여기서 코퍼스는 여섯 개의 토큰 (단어구성단위 라고 합시다 일단)으로 구성된 아주 작은 Dict를 yield했겠지요. 그렇다면 상기 문장은 \[1 0 1 0 0 1\]의 size(Dict)-dim binary vector로 표현되는 겁니다. 이렇게 해서 뭘 할 수 있냐구요? 이제 컴퓨터도 알아먹는 수치적 정보가 되었으니, 각종 분류기에 넣어 재미를 볼수 있죠! 
* 물론 태클이 들어올 수 있습니다. 저 문장을 사실 '신' '이' 그대' '-ㄹ' '사랑' 'ㅎ' '-애' 로 나눠야 합당하지 않느냐, 한 문장에서 여러 번 카운트되는 단어들이 있으면 1-hot은 부당한 representation이 아니냐 뭐 그런... 첫 번째의 경우 우리가 문장을 자소로 분리하지 않는 Twitter analyzer을 썼기 때문에 어쩔 수 없는 부분입니다. 두 번째의 경우 다음 chater에서 더 다뤄 보도록 하겠습니다.

## 3. TF-IDF and basic classifiers
Previously, we've introduced one-hot encoding of the words and the sparse sentence representation based on BoW model. However, despite its transparency and conciseness, one-hot encoding does not convey the word frequency regarding the document. 

* 앞서 bag-of-words 모델을 통해 문장을 1-hot vector들의 합(여기서 합은 boolean의 or에 해당)으로 나타내어 컴퓨터가 알아먹을 만한 어떤 수치로 나타내는 과정을 설명했습니다. 이는 전통적이고 직관적이며 상당히 강력한 방법이기도 합니다.
* 하지만 이 방법의 문제는 문장 내에 등장하는 모든 단어들을, 등장 횟수에 상관없이 공평하게 대한다는 것입니다. 예컨대 "나는 아브라함의 하나님이요 이삭의 하나님이요 야곱의 하나님이로라 하신 것을 읽어 보지 못하였느냐 (전 종교가 없습니다! 그냥 예문을 찾고싶었을뿐)" 라는 문장이 하고 싶은 말은 누가 봐도 내가 하나님이란 것 같은데 아브라함이니 이삭이니 하는 것들과 동급으로 1로 카운트되면 얼마나 억울하겠습니까. 
* 이러한 문제를 해결하기 위해 우린 일종의 normalization으로 볼 수 있는 term frequency, 즉 문장의 전체 word중 해당 word의 빈도를 고려해 문장을 벡터화해 볼 수 있습니다. 위의 문장에선 '하나님'이 3/N (형태소분석한 결과를 모두 카운트하기 귀찮으니 N으로 퉁칩시다)의 값을 부여받고 나머지 녀석들은 1/N의 값을 부여받게 되겠죠. 

But the problem is, many particles in Korean are used repetitively in the sentences, counted as 

* 그런데 또 문제가 있습니다. 바로 하나님이 소유를 나타내는 '의'와 동급이 돼버리는 겁니다. 아아, 이렇게 원통할 데가... 딱 봐도 '의'라는 녀석은 문장의 의미를 판단하는 데에 큰 도움을 주지 않을 것으로 보입니다. 다른 문장들에도 많이 나올 게 분명하거든요. 
* 그래서 우리가 생각해볼 수 있는 건 '다른 문장들에도 많이 나오는 녀석엔 가중치를 조금 주면 어떨까?'하는 것입니다. 예컨대 전체 문장 중 해당 문장이 나오는 비율의 역수 같은 걸 곱해준다면? 이런 생각으로 나온 녀석이 바로 inverse document frequency입니다. 약간의 smoothing factor을 추가하자면, test corpus에 해당 term이 없는 경우를 대비해 분모에 1을 더해주고, corpus size가 방대해질 때를 고려해 log를 입히는 정도?
* 상기한 두 개의 요소를 곱해 BoW를 개량한 모델이 바로 TF-IDF (term frequency-inverse document frequency) 입니다. 문서 분류에 아직도 활발히 사용되는 모델이지요. 

The aforementioned sentence representation can be directly utilized with the basic classifiers such as 

* 지금까지  tf-idf를 이용한 sentence 의 sparse representation에 대해 얘기했지요. 이제 구체적으로 이 녀석들을 문장 분류에 이용해먹을 만한 방법들에 대해 생각해봐야 하는데요, 그 중 하나가 이 글의 task로 제시된 분류 (classification)입니다. 
* 사실 데이터 집합(문장들)과 레이블 집합(긍정문/부정문, 의문문별 분류, 토픽별 분류 등)으로 된 트레이닝 데이터를 input으로 넣어, 별도의 테스트 데이터로 얻어진 prediction와 정답의 비교를 통해 그 네트워크의 accuracy, F-measure 등을 evaluate하여 성능을 평가한다는 점은 많은 분야에서 사용되는 분류기 evaluation의 구조입니다. 
* 이 때 accuracy는 전체 테스트 인풋 중 분류에 성공한 input의 비율이 될 겁니다. 그런데 함정이 있다면, 데이터 편중을 배제하기 어렵단 것이죠. 예컨대 데이터셋의 90퍼센트가 긍정문이고 10퍼센트만 부정문이라면, 공정한 트레이닝 및 테스트가 되었다고 하기 어렵겠죠. 그래서 binary classification에서 많이 사용되는 F-measure의 경우는 precision과 recall이라는, 각각 '1이라고 했을 때 정말 맞을 확률'과 '전체 맞은 것들 중 1이라고 했을 확률'을 생각하여 이들의 조화평균을 구해주는 방향으로 accuracy의 함정을 보정합니다. 물론 단순 조화평균이 아닌 별도 parameter을 주기도 하고, multiclass에 대해선 따로 class별 weight를 정의하기도 하지만요.
* 앞서 구한 one-hot encoding이나 TF-IDF로도 이러한 evaluation을 손쉽게 할 수 있습니다. 이 때 evaluation을 위한 prediction은 앞서 말한 10%의 test data로 하게 되지요. 위의 두 모델로도, 상당히 높은 정확도를 가진 모델을 얻을 수 있음을 알 수 있습니다. 그치만 아직도 찝찝한 점은, 컴퓨터가 단어를 세고만 있지 그 단어들이 문장 내에서, 작게는 컨텍스트에서 어떤 역할을 하는지 아무것도 모르는 것 같다는 점입니다. 

## 4. Dense word embedding and document vectors

* 여태까지 한 내용을 세줄요약해 보면</br>
1. 전산언어학은 기계로 하여금 사람 말을 잘 알아먹도록 하는 학문</br>
2. 단어 및 문장을 모델링하는 기본적인 방법으로 one-hot encoding (BoW) 및 TF-IDF가 있음</br>
3. 한국어는 교착어적 특징 때문에 어절 단위 분석보다 형태소 단위 분석이 효과적임</br>
이렇습니다. 깔끔하네요... 이외에도 수치화된 문장을 분류하는 알고리즘에는 SVM 이나 LR 등이 있다, 그 성능을 평가하는 measure에는 accuracy나 F1 score 등이 있다 ... 라는 것도 간략히 말하고 지나왔죠. 
* 그런데 이러한 방법론들은 분석해야 할 데이터가 많아지고 사전의 크기가 커질수록 computation의 문제에 당면하게 됩니다. 특히나 데이터를 몰빵해넣고 병렬연산으로 승부하는 딥러닝의 경우 더욱 그렇지요. 예컨대 30 개의 형태소로 된 문장을 벡터 sequence로 나타내어 recurrent neural network 같은 시스템으로 요약하고자 할 때, 딕셔너리 사이즈가 10만이라면, 10만 x 30이라는 무시무시한 사이즈의 행렬이 문장 하나를 표현하여 인풋으로 들어가게 되는거죠. 물론 아까 보았듯 사용하는 벡터의 크기는 조절할 수 있고, 각 벡터가 각 단어를 표현하는 것이 아주 명확하긴 하지만요.
* 이럴 때 유용하게 사용되는 개념이 2013년 태동한 word2vec, 혹은 dense low-dimension embedding 입니다. 등장 목적이 위와 같다고는 할 수 없지만, 2006년부터 촉발된 딥러닝 아키텍쳐와 맞물려 사용되며 문장 분석의 패러다임 자체를 바꾸고 있죠. Word2vec, GloVe, fastText에 대한 설명을 간략하게만이라도 해보려 합니다.

The term **Word2Vec** is very intuitive, and it converts the words, which are discrete (and were sparsely represented so far), into the numerics that are close to the continuousness. Since there are a bunch of terrific articles on the topic outside, in English, a review on word2vec and its related models are discussed mainly in Korean.

* word2vec이라는 말은 상당히 직관적입니다. 말 그대로 이산적 개념인 단어를 수치적 개념인 벡터로 바꿔 준다는 의미이죠. 사실 one-hot encoding 역시 고차원의 벡터를 만들어준다는 점을 생각하면 어폐가 있긴 합니다. 그래서 두 개념의 차이를 벡터가 sparse한지 (드문드문하게 nonzero인 성분이 있는지), 아니면 dense한지 (유클리드 공간에서처럼 빽빽이 들어차 있는지)를 차이점으로 봅니다. 물론 word2vec은 후자를 의미하죠.
* 당연한 얘기겠지만, 어떤 방식의 워드 임베딩이든, 임베딩 벡터 셋을 트레이닝하는 과정에서 어떤 원칙 (혹은 제약조건)을 주냐에 따라 결과물로 나오는 벡터들 간의 관계가 달라지게 됩니다. 예컨대 아무 제약 조건도 주지 않는 one-hot encoding의 경우, 모든 단어가 평등합니다. 아무리 비슷해 보이는 단어들이라도 1이 위치하는 엔트리가 다르다면 아무 상관없는 단어인 것이나 다름없게 되는 것이죠. distributional semantic에 대한 고려는 one-hot vector에 들어 있지 않은 겁니다.
* 모든 sparse vector가 one-hot vector같이 엔트리 간 equivalence를 지니는 건 아닙니다. 오히려 희소한 케이스에 가깝죠. 가중치를 곱해준 TF-IDF만 해도, 개별 단어를 모델링하지는 않지만 어떤 분포적인 특성이 반영되게 됩니다. binary이지만 multi-hot encoding을 차용하는 boolean distributional semantics의 경우, 의미론에서 얘기하는 feature의 개념이 등장하여, taxonomy 상 상위 위계에 해당하는 단어의 embedding이 하위 위계의 단어의 embedding에 포함되게 하는 방식으로 트레이닝이 됩니다. 아마도 벡터 사이즈는 one-hot보다는 줄어들겠죠? 벡터들 간의 distance measure가 euclidean이 아닌 어떤 이산적인 개념이 된다는 건 challenging하긴 합니다만, 언어가 기본적으로 이산적인 속성을 버릴 수 없다는 생각을 갖고 있는 저로썬 매우 매력적이라는 생각을 했습니다.

* 그렇지만 이상은 이상이고, 우리는 많은 순간 현실과 타협해야 합니다. 작은 벡터에 정보들을 우겨 넣어야 하죠.  또한 back propagation과 같은 연산을 통해 이루어지는 최신 최적화 기법들의 수혜를 받으려면, 벡터 간의 거리가 어떤 미분가능한, 이산적이지 않은 개념으로 정의가 되어야 함도 사실입니다 (물론 이산적인 목적 함수들을 위해 별도의 신경망을 연결해 주는 알고리즘도 있는 것으로 압니다만 일단 그건 나중에 생각하도록 하지요). 그러기 위해서, 고차원의 벡터를 저차원에 임베딩해 넣으면서도, 단어들 간의 관계가 좀 더 유기적으로 연결될 수 있는 방법엔 뭐가 있을까요? 
* 이러한 관점에서 볼 때, word2vec은 상당히 매력적인 시도였다 볼 수 있겠습니다. 혹자는 '비슷한 context에 등장하는 녀석들은 실제로도 비슷한/관련 있는 녀석들일 가능성이 높다는 distributional semantics의 원칙을 수치적 제약으로 잘 지키면서 one-hot vector을 저차원에 pca한 결과물'이라고 하더군요. 예컨대 '너는 나쁜 아이야'와 '너는 착한 아이야'라는 문장들을 볼 때, '나쁜'과 '착한'이 실제로 저런 류의 context를 많이 공유한다 생각해 보면, 둘이 아주 관계가 없는 단어들은 아니다, 어느정도 유기적이다, 그런 판단을 할 수 있겠죠? word2vec의 큰 철학은 이렇습니다. 
* 컨텍스트를 주고 center word를 추론하는 CBOW와 그 반대인 skip-gram (SG) 모두 word2vec의 최적화와 관련된 알고리듬인데요, SG가 여러 태스크에서 더 성능이 좋음이 보여진 바 있고 실제로도 더 자주 활용됩니다. 수식으로 나타내면 아래와 같게 되죠. 물론 이 계산을 가능하게 한 hierarchical softmax나 negative sampling 등의 개념도 무시할 수는 없지만요. 자세한 설명은 논문들에 잘 나와 있으니 이쯤 하고, 다음으로는 word2vec을 이용해 할 수 있는 게 뭔지 알아보도록 하겠습니다.

## 5. NN classifier using Keras



## 6. CNN-based sentence classification
## 7. RNN (BiLSTM)-based sentence classification
## 8. Character embedding
## 9. Concatenation of CNN and RNN layers
## 10. BiLSTM Self-attention

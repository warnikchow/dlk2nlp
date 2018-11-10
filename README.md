# DLK2NLP
### Day-by-day Line-by-line Keras-based Korean NLP
## Sentence classification: From data construction to BiLSTM self-attention
* 문장 분류 task를 중심으로 본 한국어 NLP 튜토리얼입니다. 본 튜토리얼에 사용된 컨텐츠 일부는 저자가 운영하는 [페이스북 페이지](https://www.facebook.com/nobodybelongs/notes/)에 기고한 글에서 발췌하였음을 명시합니다. 

## Contents (to be updated)
[0. Corpus labelling](https://github.com/warnikchow/dlk2nlp/blob/master/README.md#0-corpus-labeling)</br>
[1. Data preprocessing](https://github.com/warnikchow/dlk2nlp#1-data-preprocessing)</br>
[2. One-hot encoding and sentence vector](https://github.com/warnikchow/dlk2nlp#2-one-hot-encoding-and-sentence-vector)</br>
3. TF-IDF and basic classifiers
4. NN classifier using Keras
5. Dense word vector embedding and Document vectors
6. CNN-based sentence classification
7. RNN (BiLSTM)-based sentence classification
8. Character embedding
9. Concatenation of CNN and RNN layers
10. BiLSTM Self-attention

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

The fundamental of computational linguistics lies in making machines understand human utterances (or, natural language). Thus, it it 

* 전산 언어학은 기본적으로 컴퓨터한테 사람이 말하는 것, 즉 자연어를 이해시키는 과정이라고 할 수 있겠습니다. 그래서 우리는 컴퓨터로 하여금, '아 진짜 우리 말을 이해하진 못해도, 대충 어떤 내용인지 판단은 할 수 있게 해 보자' 생각을 하게 됩니다. 그렇게 해서 나오게 된 표현법 (representation) 중 가장 기본적으로 사용되고, 매우 강력한 편이라 아직까지도 많은 곳에서 사용하고 있는 방법론은 바로 단어의 1-hot encoding입니다. 
* 이 방법을 쓰기 위해선 '우리가 생각할 단어 전체 set = Dict'을 생각해야 합니다. 가장 먼저 생각해볼 수 있는 건 우리가 사용할 코퍼스에 있는 모든 단어들의 모임이죠. 코퍼스의 사이즈가 커질수록 Dict도 커질 것이고, 물론 단어는 countably infinite하게 만들어낼 수 있지만 여기선 '그나마 자주 쓰이는 녀석들'로 생각합시다. 
* 이 Dict가 생각보다 커서 실제로는 보통 많이 쓰이는 n만 단어 같이 제한을 둬서 잡습니다. 목적에 따라 functional한 녀석들은 아예 카운트하지 않기도 해요. 이 때 size(Dict) = N 이라 하면, 우리는 (고려하기로 한) 모든 단어들을 N-dim의 1-hot vector로 표현할 수 있게 되는 겁니다.

What should we do with these large-dimensional vectors? The first thing we can think of is a feature called *Bag-of-Words* (BoW). The literal meaning is a bag which contains words, and for Korean, it might be either word (*eojeol*), morpheme, characters, or alphabets (*Jamo*). Since we've decided only to use morphemes for the sparse representations, but these can be replaced with whatever feature you want to adopt. BoW approach is very simple; it just assigns 1 to the entry that corresponds with a word, 

* 이걸 갖고 뭘 하느냐? 가장 먼저 생각해볼 수 있는 것은 bag-of-words란 녀석입니다. 말 그대로 '단어가 든 가방' 이에요. 이 때 가방 = 문장 입니다. 문장 안에 어떤 단어들이 들었냐를 N-dim 1-hot vector들의 sum operation (하나만 있어도 1됨) 으로 표현하는 거죠. 
* 예컨대 "신이 그댈 사랑해"라는 문장이 있고 '신이' = \[1 0 0 0 0 0 \], '그댈' = \[0 0 1 0 0 0\], '사랑해' = \[0 0 0 0 0 1\]로 표현된다고 합시다. 여기서 코퍼스는 여섯 개의 토큰 (단어구성단위 라고 합시다 일단)으로 구성된 아주 작은 Dict를 yield했겠지요. 그렇다면 상기 문장은 \[1 0 1 0 0 1\]의 size(Dict)-dim binary vector로 표현되는 겁니다. 이렇게 해서 뭘 할 수 있냐구요? 이제 컴퓨터도 알아먹는 수치적 정보가 되었으니, 각종 분류기에 넣어 재미를 볼수 있죠! 
* 물론 태클이 들어올 수 있습니다. 저 문장을 사실 '신' '이' 그대' '-ㄹ' '사랑' 'ㅎ' '-애' 로 나눠야 합당하지 않느냐, 한 문장에서 여러 번 카운트되는 단어들이 있으면 1-hot은 부당한 representation이 아니냐 뭐 그런... 첫 번째의 경우 우리가 문장을 자소로 분리하지 않는 Twitter analyzer을 썼기 때문에 어쩔 수 없는 부분입니다. 두 번째의 경우 다음 chater에서 더 다뤄 보도록 하겠습니다.

## 3. TF-IDF and basic classifiers
## 4. NN classifier using Keras
## 5. Dense word vector embedding and Document vectors
## 6. CNN-based sentence classification
## 7. RNN (BiLSTM)-based sentence classification
## 8. Character embedding
## 9. Concatenation of CNN and RNN layers
## 10. BiLSTM Self-attention

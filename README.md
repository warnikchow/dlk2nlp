# DLK2NLP
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
8. Character embedding
9. Concatenation of CNN and RNN layers
10. BiLSTM Self-attention

## 0. Corpus labeling
The most annoying and confusing process.
Annotation guideline should be provided to annotators and more than two natives should be engaged in to make the labeling reliable and also for a computation of inter-annotator agreement (IAA). In this project, multi-class (7) annotation of short Korean utterances is utilized.
* 데이터를 만드는, 가장 귀찮은 과정입니다. 언어학적 직관은 1인분이기 때문에, 레이블링이 설득력을 얻기 위해서는 적어도 3명 이상의 1화자를 통한 레이블링으로 그 타당성을 검증해야 합니다 (아카데믹하게는...) 본 프로젝트에서는 7-class의 한국어 문장들이 분류에 사용됩니다.

The task is on classification; especially about extracting intention from a single utterance with the punctuation removed, which is suggested in [3i4k](https://github.com/warnikchow/3i4k). As the description displays, the corpus was partially hand-annotated and incorporates the utterances which are generated or semi-automatically collected. Total number of utterances reaches 57K, with each label denoting</br>
**0: Fragments**</br>
**1: Statement**</br>
**2: Question**</br>
**3: Command**</br>
**4: Rhetorical question**</br>
**5: Rhetorical command**</br>
**6: Intonation-dependent utterance**</br>
where the IAA was computed 0.85 (quite high!) for the manually annotated 2K utterance set.
* 태스크는 의도 분류로써, [3i4k](https://github.com/warnikchow/3i4k) 프로젝트를 위해 제작된 DB를 사용합니다. 사실 국책과제에 쓰려고 만든건데 어차피 논문으로도 submit했으니 공개는 상관 없지 않을까 싶어요. 5만 7천 문장쯤으로 아주 규모가 크지는 않지만, 일단 수작업으로 2만 문장 정도에서 0.85의 IAA를 얻었으며 (꽤 높은 agreement!), 4만 문장 가량이 더 수집/생성되어 그래도 어느정도 쓸만한 데이터셋이 만들어졌습니다. 레이블 7개는 위에 써 둔 것처럼, Statement~Rhetorical question까지의 clear한 의도 5가지와 (논문에선 clear-cut cases라고 칭했습니다만), 의도가 불분명한 Fragment (명사, 명사구, 혹은 불완전한 문장), 마지막으로 Intonation-dependent utterances *억양에 따라 의도가 달라지는 문형* 입니다. 마지막 레이블은 저 논문에서 하나의 레이블로 하기로 제안한 것이지만, 한국어 화자라면 어떤 문장들이 그런 성질을 가지는지 감이 올 것입니다. "뭐 먹고 싶어" "천천히 가고 있어" 같은 문장들이 그러한 유형이죠. Spoken language understanding에 아주 골머리를 썩이는 녀석들이기 때문에 따로 분류하기로 하였습니다. 
* Annotation guideline이 어떤 형식인지 궁금하신 분들은 [이곳](https://drive.google.com/open?id=1AvxzEHr7wccMw7LYh0J3Xbx5GLFfcvMW)을 참고하시면 됩니다.

## 1. Data preprocessing
For the next step, we should pay attention to how we can manage HANGUL, the letters of Korean writing system (WS). Korean WS, letter and its alphabets *Jamo* which was (maybe solely) invented by The Great King Sejong, incorporate special morpho-syllabic blocks which have a role of syllable and consist of CV(C), making Korean as a representative language with featural WS. The blocks are agglutinated to make up the word *eojeol*, which should actually be decomposed into morphemes for a semantically meaningful language processing. The spacings go between the *eojeal*s, to enhance the readability of a sentence. Many morphological analyzers give an additional spacing between the morphemes; some analyzers such as [Twitter](https://github.com/twitter/twitter-korean-text) simply give spaces, and there are the ones which conduct an elaborate tokenizing process of block decomposition, such as [Kkma](http://kkma.snu.ac.kr/). Total five analyzers are wrapped in the famous Korean natural language processing toolkit, [KoNLPy](http://konlpy.org/en/v0.4.4/)
* 한국어와 한글, 문자 체계, 형태소, 띄어쓰기 및 형태소 분석기에 대해 주절주절 설명해 봤는데요, 아마 한국어 L1 화자라면 대부분 익숙한 내용이실 테니 이 부분의 한글 설명은 생략하도록 하겠습니다 ㅎㅎ 아직 국제 무대에서는 크게 주목받지는 못하지만 그래도 한국어는 화자 수로만 봐도 세계 15위권 안에 들고 진짜로 몇 안되는 featural WS (갓종갓왕님 1인 프로젝트 덕분에 sign writing이나 소설을 위해 만들어진 언어 등과 같은 문자형식 지위를 획득 ...) 이며 promissive와 같은 독특한 particle 덕분에 언어학 서적에서도 왕왕 나오는 언어 및 문자체계를 갖고 있습니다. 또한 BTS의 떡상으로 전세계 사람들이 한국과 한국어와 한글을 알아서 Korean NLP가 국제무대에서도 Chinese나 Arabic만큼 중요하게 다뤄지는 때가 오기를... 혹은 통일을 기원하며 떡상을 기다리는...

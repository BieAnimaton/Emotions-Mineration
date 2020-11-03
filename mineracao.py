from csv import excel

import nltk

#nltk.download()

base = [('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia está muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo'),
        ('ontem eu fui roubado', 'medo'),
        ('ela me abraçou', 'alegria'),
        ('ganhei um presente super bonito no natal', 'alegria'),
        ('ele sacou a arma e saiu correndo', 'medo'),
        ('ela me deu um beijo, fiquei apaixonado', 'alegria'),
        ('me criticaram na escola', 'medo'),
        ('meu amigo teve a perna machucada', 'medo'),
        ('cai na rua e me machuquei', 'medo'),
        ('ganhei novas personagens no jogo', 'alegria'),
        ('conquistei algumas relíquias', 'alegria')]

stopwords = ['a', 'agora', 'algum', 'alguma', 'aquele', 'aqueles', 'de', 'deu', 'do', 'e', 'estou', 'esta', 'esta',
             'ir', 'meu', 'muito', 'mesmo', 'no', 'nossa', 'o', 'outro', 'para', 'que', 'sem', 'talvez', 'tem', 'tendo',
             'tenha', 'teve', 'tive', 'todo', 'um', 'uma', 'umas', 'uns', 'vou', 'um', 'uma']

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
#print(stopwordsnltk)

def removestopword(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases

#print(removestopword(base))

def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasessteming = []
    for (palavras, emocao) in texto:
        comsteming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasessteming.append((comsteming, emocao))
    return frasessteming

frasescomstemming = aplicastemmer(base)
#print(frasescomstemming)

def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavras = buscapalavras(frasescomstemming)
#print(palavras)

def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia = buscafrequencia(palavras)
#print(frequencia.most_common(50))

def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicas = buscafrequencia(frequencia)
#print(palavrasunicas)

def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicas:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

caracteristicasfrase = extratorpalavras(['am', 'nov', 'dia'])
#print(caracteristicasfrase)

basecompleta = nltk.classify.apply_features(extratorpalavras, frasescomstemming)
#print(basecompleta)

# constroi a tabela de probabilidade
classificador = nltk.NaiveBayesClassifier.train(basecompleta)
#print(classificador.labels())
#print(classificador.show_most_informative_features(5))

def linhas():
    print("-" * 50)

# FRASE SELECIONADA
teste = input(str('Digite uma frase para classificarmos sua emoção!! '))

testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavras) in teste.split():
    comstem = [p for p in palavras.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))
linhas()
print('RADICAIS: ',  testestemming)

novo = extratorpalavras(testestemming)
#print(novo)

linhas()
print('EMOÇÃO: ', classificador.classify(novo))
linhas()
distribuicao = classificador.prob_classify(novo)
for classe in distribuicao.samples():
    print("emoção |%s| com uma porcentagem de |%f|" % (classe, distribuicao.prob(classe)*100))
linhas()
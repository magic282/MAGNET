import logging
import torch
import s2s

try:
    import ipdb
except ImportError:
    pass

lower = True
seq_length = 100
report_every = 100000
shuffle = 1
MAX_LDA_WORDS = 10

logger = logging.getLogger(__name__)


def makeVocabulary(filenames, size):
    vocab = s2s.Dict([s2s.Constants.PAD_WORD, s2s.Constants.UNK_WORD,
                      s2s.Constants.BOS_WORD, s2s.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = s2s.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, ldaFile, tgtFile, srcDicts, ldaDicts, tgtDicts):
    src, tgt = [], []
    eq_mask = []
    lda = []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    ldaF = open(ldaFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')

    while True:
        sline = srcF.readline()
        ldaLine = ldaF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "" and ldaLine == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "" or ldaLine == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        ldaLine = ldaLine.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "" or ldaLine == "":
            # TODO: Fix this, does this affect dev
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        srcWords = sline.split(' ')
        ldaWords = ldaLine.split(' ')
        tgtWords = tline.split(' ')

        if len(srcWords) <= seq_length and len(tgtWords) <= seq_length:
            src += [srcDicts.convertToIdx(srcWords,
                                          s2s.Constants.UNK_WORD)]
            eq_mask += [
                torch.ByteTensor(
                    [1 if ((len(x) == 1 and 'a' <= x <= 'z') or x.startswith('[num')) else 0 for x in srcWords])]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          s2s.Constants.UNK_WORD,
                                          s2s.Constants.BOS_WORD,
                                          s2s.Constants.EOS_WORD)]
            lda += [ldaDicts.convertToIdx(ldaWords, s2s.Constants.UNK_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    ldaF.close()
    tgtF.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        eq_mask = [eq_mask[idx] for idx in perm]
        lda = [lda[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    eq_mask = [eq_mask[idx] for idx in perm]
    lda = [lda[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    return src, eq_mask, lda, tgt


def prepare_data_online(train_src, src_vocab, train_tgt, tgt_vocab, train_lda, lda_vocab):
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)
    dicts['lda'] = initVocabulary('lda', [train_lda], lda_vocab, 0)

    logger.info('Preparing training ...')
    train = {}
    train['src'], train['eq_mask'], train['lda'], train['tgt'] = makeData(train_src, train_lda, train_tgt,
                                                                          dicts['src'], dicts['lda'], dicts['tgt'])

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset

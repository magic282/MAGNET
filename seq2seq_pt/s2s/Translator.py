import s2s
import torch.nn as nn
import torch
from torch.autograd import Variable

try:
    import ipdb
except ImportError:
    pass


class Translator(object):
    def __init__(self, opt, model=None, dataset=None):
        self.opt = opt

        if model is None:

            checkpoint = torch.load(opt.model)

            model_opt = checkpoint['opt']
            self.src_dict = checkpoint['dicts']['src']
            self.lda_dict = checkpoint['dicts']['lda']
            self.tgt_dict = checkpoint['dicts']['tgt']

            self.enc_rnn_size = model_opt.enc_rnn_size
            self.dec_rnn_size = model_opt.dec_rnn_size
            encoder = s2s.Models.Encoder(model_opt, self.src_dict)
            topic_encoder = s2s.Models.TopicEncoder(model_opt, self.lda_dict)
            decoder = s2s.Models.MPGDecoder(model_opt, self.tgt_dict)
            decIniter = s2s.Models.DecInit(model_opt)
            model = s2s.Models.NMTModel(encoder, topic_encoder, decoder, decIniter)

            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size // model_opt.maxout_pool_size, self.tgt_dict.size()),
                nn.LogSoftmax())

            model.load_state_dict(checkpoint['model'])
            generator.load_state_dict(checkpoint['generator'])

            if opt.cuda:
                model.cuda()
                generator.cuda()
            else:
                model.cpu()
                generator.cpu()

            model.generator = generator
        else:
            self.src_dict = dataset['dicts']['src']
            self.tgt_dict = dataset['dicts']['tgt']
            self.lda_dict = dataset['dicts']['lda']

            self.enc_rnn_size = opt.enc_rnn_size
            self.dec_rnn_size = opt.dec_rnn_size
            self.opt.cuda = True if len(opt.gpus) >= 1 else False
            self.opt.n_best = 1
            self.opt.replace_unk = False

        self.tt = torch.cuda if opt.cuda else torch
        self.model = model
        self.model.eval()

        self.copyCount = 0

    def buildData(self, srcBatch, ldaBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                                              s2s.Constants.UNK_WORD) for b in srcBatch]
        ldaData = [self.lda_dict.convertToIdx(b, s2s.Constants.UNK_WORD) for b in ldaBatch]
        eq_mask = [
            torch.ByteTensor([1 if ((len(x) == 1 and 'a' <= x <= 'z') or x.startswith('[num')) else 0 for x in b]) for b
            in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  s2s.Constants.UNK_WORD,
                                                  s2s.Constants.BOS_WORD,
                                                  s2s.Constants.EOS_WORD) for b in goldBatch]

        return s2s.Dataset(srcData, eq_mask, ldaData, tgtData, self.opt.batch_size, self.opt.cuda)

    def buildTargetTokens(self, pred, src, attn):
        pred_word_ids = [x.item() for x in pred]
        tokens = self.tgt_dict.convertToLabels(pred_word_ids, s2s.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == s2s.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, ldaBatch, tgtBatch):
        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)
        srcBatch = srcBatch[0]  # drop the lengths needed for encoder

        topic_context, topic_mask = self.model.topic_encoder(ldaBatch)  # (seq, batch, dim)
        topic_context = topic_context.repeat(1, beamSize, 1)  # (seq, beam*batch, dim)
        # topic_att_vec = topic_context.new(topic_context.size(1), self.model.topic_encoder.size).zero_()
        # topic_att_vec.requires_grad_(False)
        topic_mask = topic_mask.unsqueeze(0).repeat(beamSize, 1, 1)

        decStates = self.model.decIniter(encStates[1])  # batch, dec_hidden

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = context.data.repeat(1, beamSize, 1)
        decStates = decStates.unsqueeze(0).data.repeat(1, beamSize, 1)
        mix_att_vec = self.model.make_init_att(context)
        padMask = srcBatch.data.eq(s2s.Constants.PAD).transpose(0, 1).unsqueeze(0).repeat(beamSize, 1, 1).float()

        beam = [s2s.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        batchIdx = list(range(batchSize))
        remainingSents = batchSize

        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)
            g_outputs, decStates, attn, \
            topic_attn, mix_att_vec, mix_gate = self.model.decoder(input, decStates, context,
                                                                   padMask.view(-1, padMask.size(2)),
                                                                   topic_context,
                                                                   topic_mask.view(-1, topic_mask.size(2)),
                                                                   mix_att_vec)

            # g_outputs: 1 x (beam*batch) x numWords
            g_outputs = g_outputs.squeeze(0)
            g_out_prob = self.model.generator.forward(g_outputs)

            # batch x beam x numWords
            wordLk = g_out_prob.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            topic_attn = topic_attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            mix_gate = mix_gate.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            father_idx = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx], topic_attn.data[idx], mix_gate.data[idx]):
                    active += [b]
                    father_idx.append(beam[b].prevKs[-1])  # this is very annoying

            if not active:
                break

            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            decStates = updateActive(decStates, self.dec_rnn_size)
            context = updateActive(context, self.enc_rnn_size)
            # att_vec = updateActive(att_vec, self.enc_rnn_size)
            padMask = padMask.index_select(1, activeIdx)
            topic_context = updateActive(topic_context, self.model.topic_encoder.size)
            mix_att_vec = updateActive(mix_att_vec, self.model.topic_encoder.size)
            topic_mask = topic_mask.index_select(1, activeIdx)

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(0, 1).contiguous()
            decStates = decStates.view(-1, decStates.size(2)).index_select(0, previous_index.view(-1)).view(
                *decStates.size())
            # att_vec = att_vec.view(-1, att_vec.size(1)).index_select(0, previous_index.view(-1)).view(*att_vec.size())
            mix_att_vec = mix_att_vec.view(-1, mix_att_vec.size(1)).index_select(0, previous_index.view(-1)).view(
                *mix_att_vec.size())

            remainingSents = len(active)

        # (4) package everything up
        allHyp, allScores, allAttn, allTopicAttn, allMixGate = [], [], [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(s2s.Constants.PAD).nonzero().squeeze(1)
            valid_topic_attn = ldaBatch[0].data[:, b].ne(s2s.Constants.PAD).nonzero().squeeze(1)
            hyps, attn, topic_attn, mix_gate = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            topic_attn = [a.index_select(1, valid_topic_attn) for a in topic_attn]
            allHyp += [hyps]
            allAttn += [attn]
            allTopicAttn += [topic_attn]
            allMixGate += [mix_gate]

        return allHyp, allScores, allAttn, allTopicAttn, allMixGate, None

    def translate(self, srcBatch, ldaBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, ldaBatch, goldBatch)
        # (wrap(srcBatch),  lengths), (wrap(tgtBatch), ), indices
        src, lda, tgt, eq_mask, indices = dataset[0]

        #  (2) translate
        pred, predScore, attn, topic_attn, mix_gate, _ = self.translateBatch(src, lda, tgt)
        pred, predScore, attn, topic_attn, mix_gate = list(zip(
            *sorted(zip(pred, predScore, attn, topic_attn, mix_gate, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [(self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n]), attn[b][n], topic_attn[b][n],
                  mix_gate[b][n]) for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, None

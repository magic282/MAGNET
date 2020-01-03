import torch
import torch.nn as nn
from torch.autograd import Variable
import s2s.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F

try:
    import ipdb
except ImportError:
    pass


class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.enc_rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=opt.layers,
                          dropout=opt.dropout,
                          bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths)
        """
        lengths = input[-1].data.view(-1).tolist()  # lengths data is wrapped inside a Variable
        wordEmb = self.word_lut(input[0])
        emb = pack(wordEmb, lengths)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class TopicEncoder(nn.Module):
    def __init__(self, opt, dicts):
        self.size = opt.word_vec_size

        super(TopicEncoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        """
        input: (wrap(ldaBatch), lda_length)
        """
        mask = input[0].eq(s2s.Constants.PAD).float().transpose(0, 1).contiguous()  # (batch, seq)
        wordEmb = self.word_lut(input[0])
        return wordEmb, mask


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class Decoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size + opt.word_vec_size), opt.dec_rnn_size)
        self.maxout = s2s.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.hidden_size = opt.dec_rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, src_pad_mask, init_att):
        emb = self.word_lut(input)

        g_outputs = []
        cur_context = init_att
        self.attn.applyMask(src_pad_mask)
        precompute = None
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context], 1)
            output, hidden = self.rnn(input_emb, hidden)
            cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

            readout = self.readout(torch.cat((emb_t, output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
        g_outputs = torch.stack(g_outputs)
        return g_outputs, hidden, attn, cur_context


class MPGDecoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size

        super(MPGDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.topic_attn = s2s.modules.ConcatAttention(opt.word_vec_size, opt.dec_rnn_size, opt.att_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size + opt.word_vec_size + opt.dec_rnn_size),
                                 opt.dec_rnn_size)
        self.mix_gate = nn.Linear(opt.dec_rnn_size, 1)
        self.maxout = s2s.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.hidden_size = opt.dec_rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, src_pad_mask, topic_context, topic_mask, mix_init_context):
        emb = self.word_lut(input)

        g_outputs = []
        all_attn = []
        all_topic_attn = []
        all_gate = []
        # cur_context = init_att
        # cur_topic_context = topic_init_att
        cur_mix_context = mix_init_context
        self.attn.applyMask(src_pad_mask)
        self.topic_attn.applyMask(topic_mask)
        precompute = None
        topic_precompute = None
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_mix_context], 1)
            output, hidden = self.rnn(input_emb, hidden)
            mix_gate_value = F.sigmoid(self.mix_gate(output))

            cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)
            cur_topic_context, topic_attn, topic_precompute = self.topic_attn(output, topic_context.transpose(0, 1),
                                                                              topic_precompute)
            cur_mix_context = mix_gate_value * cur_context + (1 - mix_gate_value) * cur_topic_context
            all_attn.append(attn)
            all_topic_attn.append(topic_attn)
            all_gate.append(mix_gate_value)

            readout = self.readout(torch.cat((emb_t, output, cur_mix_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
        g_outputs = torch.stack(g_outputs)
        attn = torch.stack(all_attn)
        topic_attn = torch.stack(all_topic_attn)
        gate_values = torch.stack(all_gate)
        return g_outputs, hidden, attn, topic_attn, cur_mix_context, gate_values


class DecInit(nn.Module):
    def __init__(self, opt):
        super(DecInit, self).__init__()
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = opt.enc_rnn_size
        self.dec_rnn_size = opt.dec_rnn_size
        self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)
        self.tanh = nn.Tanh()

    def forward(self, last_enc_h):
        # batchSize = last_enc_h.size(0)
        # dim = last_enc_h.size(1)
        return self.tanh(self.initer(last_enc_h))


class NMTModel(nn.Module):
    def __init__(self, encoder: Encoder,
                 topic_encoder: TopicEncoder,
                 decoder: MPGDecoder,
                 decIniter: DecInit):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.topic_encoder = topic_encoder
        self.decoder = decoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, input):
        """
        input: (wrap(srcBatch), lengths), (wrap(ldaBatch), lda_length), (wrap(tgtBatch),), indices
        """
        # ipdb.set_trace()
        src = input[0]
        tgt = input[2][0][:-1]  # exclude last target from inputs
        src_pad_mask = Variable(src[0].data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        enc_hidden, context = self.encoder(src)

        topic_context, topic_mask = self.topic_encoder(input[1])
        # topic_init_att = topic_context.new(topic_context.size(1), self.topic_encoder.size).zero_()
        # topic_init_att.requires_grad_(False)

        init_att = self.make_init_att(context)
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden

        # return g_outputs, hidden, attn, topic_attn, cur_context, cur_topic_context
        g_out, dec_hidden, _attn, _topic_attn, \
        _mix_attention_vector, gate_values, = self.decoder(tgt, enc_hidden, context, src_pad_mask,
                                             topic_context, topic_mask, init_att)

        return g_out, _attn, _topic_attn, src_pad_mask, gate_values

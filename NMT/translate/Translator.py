import torch
from torch.autograd import Variable
import torch.nn.functional as F
from NMT.translate.Beam import Beam

from Utils.DataLoader import PAD_WORD
from Utils.DataLoader import BOS_WORD
from Utils.DataLoader import EOS_WORD


from .Translation import TranslationBuilder



class BatchTranslator(object):
    """
    Uses a model to translate a batch of sentences.

    Args:
       model (:obj:`NMT.Models`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       k_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
    """

    def __init__(self, model, config, src_vocab, trg_vocab):
        self.config = config
        self.vocab = trg_vocab
        self.model = model 
        self.k_best = config.k_best
        self.max_length = config.max_length
        self.beam_size = config.beam_size
        self.stepwise_penalty = config.stepwise_penalty
        self.PAD_WID = trg_vocab.stoi[PAD_WORD]
        self.BOS_WID = trg_vocab.stoi[BOS_WORD]
        self.EOS_WID = trg_vocab.stoi[EOS_WORD]
        self.use_beam_search = config.use_beam_search
        self.builder = TranslationBuilder(src_vocab, trg_vocab, config)
    def beam_search_decoding(self, encoder_output, src_length, decoder_state):
        """
        beam search. 

        Args:
           encoder_output (`Variable`): the output of encoder hidden layer [L_s, B, H]
           decoder_state (`Variable`): the state of encoder  
        """
        batch_size = encoder_output.size(1)
        beam = [Beam(self.beam_size, self.PAD_WID, self.BOS_WID, self.EOS_WID,
            n_best=self.k_best) for _ in range(batch_size)]
       

        # repeat source `beam_size` times.
        # [seq_len x beam_size x batch_size x ?]
        # [L_s, B, H] -> [L_s, K x B, H]
        encoder_outputs = Variable(
            encoder_output.data.repeat(1, self.beam_size, 1))
        src_lengths = src_length.repeat(self.beam_size)
        decoder_state.repeat_beam_size_times(self.beam_size)
        
        for i in range(self.max_length):
            with torch.no_grad():
                if all((b.done() for b in beam)):
                    break
                cads = [b.get_current() for b in beam]
                trg = torch.stack(cads).view(1, -1)

                outputs, decoder_state, attn = self.model.decode(
                    trg, encoder_outputs, src_lengths, decoder_state)[:3]
                
                dist = F.log_softmax(
                    self.model.generator(outputs), dim=-1)

                score, idx = dist.topk(1, dim=-1)  
                
                
                def unpack(x):
                    return x.view(self.beam_size, batch_size, -1).contiguous()
                outputs = unpack(dist.squeeze(0))
                beam_attn = unpack(attn)


                for j, b in enumerate(beam):
                    # j: batch_size
                    b.advance(outputs.data[:, j, :], 
                        beam_attn.data[:, j, :src_lengths[j]])
                    decoder_state.beam_update_state(j, b.get_origin())
                del dist, outputs
        return beam

    def monotonic_decoding(self, encoder_outputs, src_length, state):
        """
        beam search. 

        Args:
           encoder_output (`Variable`): the output of encoder hidden layer [L_s, B, H]
           decoder_state (`Variable`): the state of encoder  
        """
        
        batch_size = encoder_outputs.size(1)
        attns = []
        preds = []
        scores = []
        trg = torch.LongTensor(batch_size).fill_(self.BOS_WID).cuda()
        scores = torch.FloatTensor(batch_size).fill_(0).cuda()
        #print([self.vocab.itos[x] for x in trg.tolist()])
            
        for i in range(self.max_length):
            with torch.no_grad():
                output, attn, state = self.model.translate_step(
                    trg.unsqueeze(0), encoder_outputs, src_length, state)
                
                dist = F.log_softmax(
                            self.model.generator(output), 
                            dim=-1)
                
                score, idx = dist.topk(1, dim=-1)  
                del dist, output
                trg, score = idx.squeeze(1), score.squeeze(1)
                score.masked_fill_(trg.eq(self.PAD_WID), 0).float()
                scores += score.data
                preds.append(trg)
                attns.append(attn)

        preds = torch.stack(preds, dim=0)\
                        .transpose(0, 1)\
                        .unsqueeze(1).contiguous()
        
        attns = torch.stack(attns, dim=0)\
                        .transpose(0, 1)\
                        .unsqueeze(1).contiguous()

        scores = scores.unsqueeze(1).contiguous()
        return preds, scores, attns
    
    


    def translate(self, batch):
        """
        Translate a batch of sentences. 

        Args:
           batch (Batch): a batch from a dataset object
        """
        
        # 1. encoding
        src, src_length = batch.src, batch.src_Ls

        encoder_outputs, encoder_state = \
                self.model.encoder(src, src_length)

        # 2. encoder to decoder
        decoder_state = self.model.decoder.init_decoder_state(encoder_state)

        # 3. generate translations using beam search.
        if self.use_beam_search:
            beam = self.beam_search_decoding(
                 encoder_outputs, src_length, decoder_state)
            preds, scores, attns = self.extract_from_beam(beam)
        else:
            preds, scores, attns = self.monotonic_decoding(
                encoder_outputs, src_length, decoder_state)
        
        del encoder_outputs, encoder_state

        gold_scores = self.get_gold_scores(batch)
        batch_trans = self.builder.build(batch, preds, scores, attns, gold_scores)
        
        # print([self.vocab.itos[i] for i in src[:,0].tolist()])
        # print([self.vocab.itos[i] for i in preds.squeeze().tolist()])
        # print([self.vocab.itos[i] for i in batch.trg[:,0].tolist()])
        return batch_trans

    
    def extract_from_beam(self, beam):
        """
        extract translations from beam.
        """
        preds = [[]]
        scores = [[]]
        attns = [[]]
        for b in beam:
            best_k = b.sort_finished(minimum=self.k_best)[:self.k_best]
            for score, t, k in best_k:
                wid, attn = b.get_hypo(t, k)
                preds[-1].append(torch.IntTensor(wid))
                scores[-1].append(score)
                attns[-1].append(attn)
            preds.append([])
            scores.append([])
            attns.append([])
        return preds[:-1], scores[:-1], attns[:-1]

    def get_gold_scores(self, batch):
        src = batch.src
        src_lengths = batch.src_Ls

        trg_in = batch.trg[:-1]
        trg_out = batch.trg[1:]

        encoder_outputs, encoder_state = \
            self.model.encoder(src, src_lengths)

        decoder_state = self.model.decoder.init_decoder_state(encoder_state)

        gold_scores = torch.FloatTensor(batch.batch_size).fill_(0).cuda()

     
        for t_in, t_out in zip(trg_in.split(1, dim=0), trg_out.split(1, dim=0)):
            output = self.model.translate_step(
                t_in, encoder_outputs, src_lengths, decoder_state)[0]

            scores = F.log_softmax(
                self.model.generator(output), dim=-1)
            
            trg = t_out.transpose(0, 1)
            scores = scores.data.gather(1, trg)
            scores.masked_fill_(trg.eq(self.PAD_WID), 0).float()
            gold_scores += scores.data.squeeze(1)
        return gold_scores

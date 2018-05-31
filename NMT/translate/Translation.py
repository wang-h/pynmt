import torch
from Utils.DataLoader import EOS_WORD
from Utils.DataLoader import UNK_WORD
from Utils.DataLoader import PAD_WORD
from Utils.DataLoader import BOS_WORD
from Utils.log import trace


class TranslationBuilder(object):
    """
    Luong et al, 2015. Addressing the Rare Word Problem in Neural Machine Translation.
    """

    def __init__(self, src_vocab, trg_vocab, config):
        """
        Args:
        vocab (Vocab): vocabulary
        replace_unk (bool): replace unknown words using attention
        """
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.replace_unk = config.replace_unk
        self.k_best = config.k_best

    def _build_sentence(self, vocab, pred, src, attn=None):
        """
        build sentence using predicted output with the given vocabulary.
        """
        tokens = []
        for wid in pred:
            token = vocab.itos[int(wid)]
            if token == BOS_WORD:
                continue
            if token == EOS_WORD:
                break
            tokens.append(token)

        if self.replace_unk and (attn is not None) and (src is not None):
            for i in range(len(tokens)):
                if tokens[i] == UNK_WORD:
                    _, max_ = attn[i].max(0)
                    tokens[i] = self.src_vocab.itos[src[int(max_)]]
                    
        return tokens

    def build_target(self, pred, src, attn=None):

        return self._build_sentence(
            self.trg_vocab, pred, src, attn)

    def build_source(self, src):

        return self._build_sentence(
            self.src_vocab, src, src)

    def build(self, batch, preds, scores, attns, gold_scores=None):
        """
        build translation from batch output 
        Args:
            preds : `[B  x K_best x L_t]`.
            scores : `[B  x K_best]`.
            attns : `[B  x K_best x L_t x L_s]`.
        """
        batch_size = batch.batch_size

        translations = [None] * batch_size
        order =  batch.sid % batch_size

        
        for i in range(batch_size):
            src = batch.src[:, i].tolist()
            input_sent = self.build_source(src)
            pred_sents = []
            for k in range(self.k_best):
                sent = self.build_target(preds[i][k], src, attns[i][k])
                pred_sents.append(sent)
            if batch.trg is not None:
                gold = batch.trg[:, i].tolist()
                gold_sent = self.build_target(gold, src)

            translations[order[i]] = Translation(
                    input_sent, pred_sents,
                    attns[i], scores[i],
                    gold_sent, gold_scores[i])
        return translations

class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention distributions for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, src, preds, attns, pred_scores, gold, gold_score):
        self.src = src
        self.preds = preds
        self.attns = attns
        self.pred_scores = pred_scores
        self.gold = gold
        self.gold_score = gold_score

    def pprint(self, sid):
        """
        Log translation to stderr.
        """
        output = '\nINPUT [{}]: {}\n'.format(sid, " ".join(self.src))

        best = self.preds[0]
        best_score = self.pred_scores[0].sum().float()
        output += 'PRED  [{}]: {}\t'.format(sid, " ".join(best))
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold is not None:
            output += 'GOLD  [{}]: {}\t'.format(sid,  ' '.join(self.gold))
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))

        return output

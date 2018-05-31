import torch
from NMT.translate import Penalties
from Utils.DataLoader import EOS_WORD

# class Beam(object):
#     """
#     Args:
#        size (int): beam size
#        pad, bos, eos (int): indices of padding, beginning, and ending.
#        k_best (int): nbest size to use
#        global_scorer (:obj:`GlobalScorer`)
#     """
#     def __init__(self, config, pad, bos, eos, global_scorer=None):

#         self.beam_size = config.beam_size
#         # The score for each translation on the beam.
#         self.scores = torch.FloatTensor(self.beam_size)\
#                             .zero_().cuda()
        
#         self.all_scores = []

#         # The backpointers at each time-step.
#         self.prev_ks = []

#         # The outputs at each time-step.
#         self.next_ys = [torch.LongTensor(self.beam_size)
#                         .fill_(bos).cuda()]
#         #self.next_ys[0] 

#         # Has EOS topped the beam yet.
#         self.eos = eos
        
#         self.eos_top = False

#         self.bos = bos

#         # The attentions (matrix) for each time.
#         self.attn = []

#         # Time and k pair for finished.
#         self.finished = []


#         # Minimum prediction length
#         self.min_length = 3

#         # Apply Penalty at every step
#         self.stepwise_penalty = config.stepwise_penalty
#         self.block_ngram_repeat = 1
        
#     def get_current(self):
#         "Get the outputs for the current timestep."
#         return self.next_ys[-1]

#     def get_previous(self):
#         "Get the backpointers for the current timestep."
#         return self.prev_ks[-1]

#     def sterilize_eos(self, current_scores):
#         # for b in beam
#         # Don't let EOS have children.
#         last_y = self.next_ys[-1]
#         for i in range(last_y.size(0)):
#             if last_y[i] == self.eos:
#                 current_scores[i] = -1e20

#     def check_finished(self):
#         for i in range(self.next_ys[-1].size(0)):
#             if self.next_ys[-1][i] == self.eos:
#                 val = self.scores[i]
#                 self.finished.append((val, len(self.next_ys) - 1, i))
                
#         # End condition is when top-of-beam is EOS and no global score.
#         if self.next_ys[-1][0] == self.eos:
#             self.all_scores.append(self.scores)
#             self.eos_top = True   

#     def check_ngram_repeat(self, current_scores):
#         if self.block_ngram_repeat > 0:
#             ngrams = []
#             le = len(self.next_ys)
#             for j in range(self.next_ys[-1].size(0)):
#                 hyp, _ = self.get_hypo(le-1, j)
#                 ngrams = set()
#                 fail = False
#                 gram = []
#                 for i in range(le-1):
#                     # Last n tokens, n = block_ngram_repeat
#                     gram = (gram + [hyp[i]])[-self.block_ngram_repeat:]
#                     # Skip the blocking if it is in the exclusion list
#                     if tuple(gram) in ngrams:
#                         fail = True
#                         ngrams.add(tuple(gram))
#                 if fail:
#                     current_scores[j] = -10e20
            
#     def advance(self, probs, attn):
#         """
#         Given prob over words for every last beam `wordLk` and attention
#         `attn_out`: Compute and update the beam search.

#         Args:

#             probs: probs of advancing from the last step [ K x V ]
#             attn: attention at the last step [ K x L_s ]

#         Returns: True if beam search is complete.
#         """
#         vocab_size = probs.size(1)
#         # increamental
#         if len(self.prev_ks) < 1:        # y_0 
#             current_scores = probs
#         else:                            # y_1, ..., y_t
#             current_scores = probs + \
#                 self.scores.unsqueeze(1).expand_as(probs)
#             self.check_ngram_repeat(current_scores) 
#             self.sterilize_eos(current_scores)
#         # flattened B x V
#         top_v, top_i = current_scores.contiguous()\
#                                      .view(-1)\
#                                      .topk(self.beam_size, 0)

#         self.scores = top_v
#         print(self.scores.size())
#         self.all_scores.append(self.scores)

#         # expand hypotheses
#         prev_k = top_i / vocab_size
#         next_y = top_i - prev_k * vocab_size
        
#         #print(self.prev_ks[-1].index_select(prev_k), next_y)
#         self.prev_ks.append(prev_k)
#         self.next_ys.append(next_y) 
#         self.attn.append(attn.index_select(0, prev_k))

        
        

        
#         self.check_finished()


#     def done(self, k_best):
#         return self.eos_top and len(self.finished) >= k_best

#     def sort_finished(self, minimum=None):
#         if minimum is not None:
#             i = 0
#             # Add from beam until we have minimum outputs.
#             while len(self.finished) < minimum:
#                 val = self.scores[i]
#                 self.finished.append((val, len(self.next_ys) - 1, i))
#                 i += 1
#         self.finished.sort(key=lambda x: -x[0])
#         return self.finished

#     def get_hypo(self, timestep, k):
#         """
#         Walk back to construct the full hypothesis.
#         """
#         hyp, attn = [], []
#         for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
#             hyp.append(self.next_ys[j+1][k])
#             attn.append(self.attn[j][k])
#             k = self.prev_ks[j][k]
#         return hyp[::-1], torch.stack(attn[::-1])


# class GNMTGlobalScorer(object):
#     """
#     NMT re-ranking score from
#     "Google's Neural Machine Translation System" :cite:`wu2016google`

#     Args:
#        alpha (float): length parameter
#        beta (float):  coverage parameter
#     """
#     def __init__(self, alpha, beta, cov_penalty, length_penalty):
#         self.alpha = alpha
#         self.beta = beta
#         penalty_builder = Penalties.PenaltyBuilder(cov_penalty,
#                                                    length_penalty)
#         # Term will be subtracted from probability
#         self.cov_penalty = penalty_builder.coverage_penalty()
#         # Probability will be divided by this
#         self.length_penalty = penalty_builder.length_penalty()

#     def score(self, beam, logprobs):
#         """
#         Rescores a prediction based on penalty functions
#         """
#         normalized_probs = self.length_penalty(beam,
#                                                logprobs,
#                                                self.alpha)
#         if not beam.stepwise_penalty:
#             penalty = self.cov_penalty(beam,
#                                        beam.global_state["coverage"],
#                                        self.beta)
#             normalized_probs -= penalty

#         return normalized_probs

#     def update_score(self, beam, attn):
#         """
#         Function to update scores of a Beam that is not finished
#         """
#         if "prev_penalty" in beam.global_state.keys():
#             beam.scores.add_(beam.global_state["prev_penalty"])
#             penalty = self.cov_penalty(beam,
#                                        beam.global_state["coverage"] + attn,
#                                        self.beta)
#             beam.scores.sub_(penalty)

#     def update_global_state(self, beam):
#         "Keeps the coverage vector as sum of attentions"
#         if len(beam.prev_ks) == 1:
#             beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
#             beam.global_state["coverage"] = beam.attn[-1]
#             self.cov_total = beam.attn[-1].sum(1)
#         else:
#             self.cov_total += torch.min(beam.attn[-1],
#                                         beam.global_state['coverage']).sum(1)
#             beam.global_state["coverage"] = beam.global_state["coverage"] \
#                 .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

#             prev_penalty = self.cov_penalty(beam,
#                                             beam.global_state["coverage"],
#                                             self.beta)
#             beam.global_state["prev_penalty"] = prev_penalty


class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """
    def __init__(self, size, pad, bos, eos,
                 n_best=1, 
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):

        self.size = size
        self.tt = torch.cuda

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        #self.global_scorer = global_scorer
        #self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    def get_current(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        # if self.stepwise_penalty:
        #     self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            # Block ngram repeats
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hypo(le-1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le-1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram + [hyp[i]])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20
        else:
            #print(word_probs.size())
            beam_scores = word_probs
            # beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.contiguous().view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        
        self.attn.append(attn_out.index_select(0, prev_k))
        #self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                # global_scores = self.global_scorer.score(self, self.scores)
                # s = global_scores[i]
                s =  self.scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                #global_scores = self.global_scorer.score(self, self.scores)
                #s = global_scores[i]
                s = self.scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        # scores = [sc for sc, _, _ in self.finished]
        # ks = [(t, k) for _, t, k in self.finished]
        # return scores, ks
        return self.finished
    def get_hypo(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])
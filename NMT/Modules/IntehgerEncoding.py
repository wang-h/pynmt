import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# the neural network model
###############################################################################

def sample_gumbel(shape, eps=1e-20):
    mu = torch.rand(shape).cuda()
    return Variable(-torch.log(-torch.log(mu + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1.0):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def softmax2onehot(logits):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """

    y = F.softmax(logits, dim=-1)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class BinaryEncoding(nn.Module):
    def __init__(self, embedding_dim, m, k, gpu=False):
        # m is the number of codebooks
        # k is the number of candidate vectors in each codebooks
        # the number of neurons in the hidden layer
        hidden_size = int(m * k / 2)
        self.package_num = m
        self.embedding_dim = embedding_dim
        super(Coding, self).__init__()
        self.bottleneck = nn.Sequential(
                                nn.Linear(embedding_dim, hidden_size),
                                nn.Tanh())
        
        self.decomposition = nn.ModuleList(
             [nn.Conv1d(embedding_dim, k, 1) for _ in range(m)])
        self.decomposition = nn.ModuleList(
            [nn.Linear(hidden_size, k) for _ in range(m)])
        self.composition = nn.ModuleList(
            [nn.Linear(k, embedding_dim, bias=False) for _ in range(m)])
        #self.fc = nn.Conv1d(32, 1, 1)
        if gpu:
            self.decomposition.cuda()
            self.composition.cuda()
        self.params_init()

    def params_init(self):
        if self.training:
            print("Initializing model parameters.", file=sys.stderr)
            for p in self.parameters():
                p.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # decomposition
        context = self.encode(input)
        #print([c.size() for c in context])
        # composition
        decoder_output = self.decode(context)
        return decoder_output

    def encode(self, input):
        # hidden layer
        h = self.bottleneck(input)
        #h = self.c1(input)
        #h = input.unsqueeze(2)
        #print(h.size())
        # decomposition
        ds = []
        for i, block in enumerate(self.decomposition):
            #h_i = block(h).squeeze(2)
            h_i = block(h)
            a_i = F.softplus(h_i)
            #d_i = gumbel_softmax(a_i)
            d_i = softmax2onehot(a_i)
            
            ds.append(d_i)
        return ds

    def decode(self, context):
        output = []
        for i, block in enumerate(self.composition):
            # print(encoder_output[i].size())
            # print(self.composition[i].size())
            output.append(block(context[i]))
        output = torch.stack(output)
        #output = self.fc(output.transpose(0,1)).squeeze(1)
        #print(output.size())
        output = torch.sum(output, dim=0)
        return output

    def hard_encode(self, input):
        """
        This function maps continuous word embeddings to integer word embeddings.
        :param input: each row is a word embedding vector
        :return: each row is the corresponding integer embedding vector
        >>> s = torch.manual_seed(1)
        >>> net = Coding(5, 3, 4) # m=3 is the dimension of integer vector
        >>> v1, v2 = list(net.get_integer_embed(Variable(torch.FloatTensor([[1,2,3,4,5],[2,3,5,7,1]]))))
        >>> list(v1)
        [1.0, 2.0, 2.0]
        >>> list(v2)
        [1.0, 2.0, 2.0]
        """
        ds = self.encode(input)
        int_vec = [d.max(dim=0)[1].data.item() for d in ds]
        return int_vec

# data
#########################################################################


class Dataset(list):
    def __init__(self, vec_path, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.load_embedding(vec_path)
        self.word_ids = {w: i for i, (w, _) in enumerate(self)}
        self.words = [w for (w, _) in self]

    def load_embedding(self, vec_path):
        with open(vec_path, "r", errors="replace") as vec_fin:
            for i, line in enumerate(vec_fin):
                if i == 0:
                    self.voc_size, self.dimension = map(
                        lambda x: int(x), line.rstrip("\n").split())
                    continue
                cols = line.rstrip("\n").split()
                w = cols[0]
                v = torch.FloatTensor([float(x) for x in cols[1:]])
                self.append((w, v))
        vec_fin.close()


# training
##########################################################################
def train(optimizer, loss_function, model, trainloader, epoch, is_cuda=False):
    print('Training on epoch No.%s' % (epoch))
    model.train()

    loss = 0.
    total = 0
    batch_size = trainloader.batch_size

    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        words, inputs = data
        targets = inputs.clone()
        cur_size = inputs.size(0)
        inputs, targets = Variable(inputs), Variable(targets)
        if is_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        

        loss = loss_function(outputs, targets)
        loss.backward()
        #print (model.decomposition[0].weight)
        optimizer.step()
        if i != 0 and i % 50 == 0:
            print('epoch No.%s , loss:%0.2f' %
                  (epoch, float(loss.data) / cur_size))
            print(words[0], model.hard_encode(outputs[0]))

# writing integer vector model
##########################################################################


def dump_int_vec(model, dataset, batch_size, is_cuda=False):
    array = []
    pbar = tqdm.tqdm(range(math.ceil(dataset.voc_size / batch_size)))
    for i in pbar:
        pbar.set_description('producing integer word embedding model')
        start = i * batch_size
        stop = (i + 1) * batch_size
        if stop > dataset.voc_size:
            stop = dataset.voc_size
        if is_cuda:
            new_matrix = model.hard_encode(
                Variable(torch.Tensor(dataset[start:stop])).cuda())
        else:
            new_matrix = model.hard_encode(
                Variable(torch.Tensor(dataset[start:stop])))

        for vec in new_matrix:
            array.append(list(vec))
    dataset.save_new_model(np.array(array))


# argument parser
################################################################
def parse_args():
    usage = '\n1. You can query by using a command as follows:\n' \
            'python manager.py -q /path/to/model/dirs -w 5 -d 100\n'

    parser = argparse.ArgumentParser(
                        description='description:\nThis Python2 program helps to manage model.\n Current version support only querying.', usage=usage)
    parser.add_argument('-m', type=int, default=32,
                        help='number of codebooks')
    parser.add_argument('-k', type=int, default=16,
                        help='number of vectors in each code book')
    parser.add_argument('--vec', type=str,
                        default="/itigo/Uploads/WMT2018/en-tr/orig_clean/ft.embed.vec",
                        help='path to word embedding file')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help="train on gpu")
    parser.add_argument('--epoch', type=int, default=20,
                        help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="number of epochs")
    args = parser.parse_args()
    return args


# main function
##########################################################################
def main(vec, m_codebooks, k, epoch, gpu, batch_size):
    """
    >>> main('/itigo/Uploads/Hongyz/WVs/088/088', True, 32, 16, 3, True)


    """

    dataset = Dataset(vec)

    loss_function = nn.MSELoss(size_average=False)
    code = Coding(dataset.dimension, m_codebooks, k, gpu)
    print (code)
    optimizer = optim.Adam(code.parameters(), lr=0.001)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    if gpu:
        code.cuda()
    for e in range(epoch):
        train(optimizer, loss_function, code, trainloader, e, gpu)
        if e == 10:
            checkpoint = {
                'model': code.state_dict(),
            }
            torch.save(checkpoint, 'code_e%d.pt' % e)

    dump_int_vec(code, dataset, batch_size, gpu)


if __name__ == "__main__":
    args = parse_args()
    main(args.vec, args.m, args.k,
         args.epoch, args.gpu, args.batch_size)

